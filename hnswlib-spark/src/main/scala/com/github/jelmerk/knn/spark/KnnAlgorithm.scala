package com.github.jelmerk.knn.spark

import java.net.InetAddress

import scala.math.abs
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.github.jelmerk.knn.scalalike._
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction

/**
  * Item in an nearest neighbor search index
  *
  * @param id item identifier
  * @param vector item vector
  */
case class IndexItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]]

/**
  * Neighbor of an item
  *
  * @param neighbor identifies the neighbor
  * @param distance distance to the item
  */
case class Neighbor(neighbor: String, distance: Float)


object Udfs {

  /**
    * Convert a dense vector to a float array.
    */
  val vectorToFloatArray: UserDefinedFunction = udf { vector: Vector => vector.toArray.map(_.toFloat) }

  /**
    * Convert a double array to a float array
    */
  val doubleArrayToFloatArray: UserDefinedFunction = udf { vector: Seq[Double] => vector.map(_.toFloat) }
}

trait KnnModelParams extends Params {

  /**
    * Param for the column name for the row identifier.
    * Default: "id"
    *
    * @group param
    */
  val identifierCol = new Param[String](this, "identifierCol", "column names for the row identifier")

  /** @group getParam */
  def getIdentifierCol: String = $(identifierCol)

  /**
    * Param for the column name for the vector.
    * Default: "vector"
    *
    * @group param
    */
  val vectorCol = new Param[String](this, "vectorCol", "column names for the vector")

  /** @group getParam */
  def getVectorCol: String = $(vectorCol)

  /**
    * Param for the column name for returned neighbors.
    * Default: "neighbors"
    *
    * @group param
    */
  val neighborsCol = new Param[String](this, "neighborsCol", "column names for returned neighbors")

  /** @group getParam */
  def getNeighborsCol: String = $(neighborsCol)

  /**
    * Param for the distance function to use. One of "bray-curtis", "canberra",  "cosine", "correlation", "euclidean", "inner-product", "manhattan"
    * Default: "cosine"
    *
    * @group param
    */
  val distanceFunction = new Param[String](this, "distanceFunction", "column names for returned neighbors")

  /** @group getParam */
  def getDistanceFunction: String = $(distanceFunction)

  /**
    * Param for number of neighbors to find (> 0).
    * Default: 5
    *
    * @group param
    */
  val k = new IntParam(this, "k", "number of neighbors to find", ParamValidators.gt(0))

  /**
    * Number of results to return as part of the knn search.
    *
    * @group getParam
    * */
  def getK: Int = $(k)

  /**
    * Param for whether or not to exclude the query row_id.
    * Default: false
    *
    * @group param
    */
  val excludeSelf = new BooleanParam(this, "excludeSelf", "whether or not to exclude the query row_id")

  setDefault(excludeSelf, false)

  /**
    * Number of results to return as part of the knn search.
    *
    * @group getParam
    * */
  def getExcludeSelf: Boolean = $(excludeSelf)

}

trait KnnAlgorithmParams extends KnnModelParams {

  /**
    * Number of partitions (default: 1)
    */
  val numPartitions = new IntParam(this, "numPartitions",
    "number of partitions", ParamValidators.gt(0))

  /** @group getParam */
  def getNumPartitions: Int = $(numPartitions)

}


/**
  * Base class for nearest neighbor search models.
  *
  * @param uid identifier
  * @param numPartitions how many partitions
  * @param partitioner the partitioner used to parition the data
  * @param indices rdd that holds the indices that are used to do the search
  * @tparam TModel model type
  */
abstract class KnnModel[TModel <: Model[TModel]](override val uid: String,
                                                  numPartitions: Int,
                                                  partitioner: Partitioner,
                                                  indices: RDD[(Int, Index[String, Array[Float], IndexItem, Float])])
  extends Model[TModel] with KnnModelParams {

  import com.github.jelmerk.knn.spark.Udfs._

  /** @group setParam */
  def setIdentifierCol(value: String): this.type = set(identifierCol, value)

  /** @group setParam */
  def setNeighborsCol(value: String): this.type = set(neighborsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setExcludeSelf(value: Boolean): this.type = set(excludeSelf, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    val identifierType = dataset.schema(getIdentifierCol).dataType

    val vectorCol = dataset.schema(getVectorCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getVectorCol)) // VectorUDT is not accessible
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getVectorCol))
      case _ => col(getVectorCol)
    }

    dataset.select(col(getIdentifierCol).cast(StringType), vectorCol.as(getVectorCol))
      .withColumn("partition", explode(array(0 until numPartitions map lit: _*)))
      .as[(String, Array[Float], Int)]
      .rdd
      .map { case (id, vector, partition) => (partition, (id, vector)) }
      .partitionBy(partitioner)
      .cogroup(indices, partitioner)
      .flatMap { case (_, (itemsIter, indicesIter)) =>
        indicesIter.headOption.map { index =>
          itemsIter.map { case (id, vector) =>

            val k =
              if (getExcludeSelf) getK
              else getK + 1

            val neighbors = index.findNearest(vector, k)
              .collect { case SearchResult(item, distance)
                if !getExcludeSelf || item.id != id => Neighbor(item.id, distance) }
              .take(getK)
            id -> neighbors

          }
        }.getOrElse(Iterator.empty)
      }
      .groupBy(_._1)
      .map { case (id, iterator) =>
        val closestNeighbors = iterator.flatMap(_._2).toList.sortBy(_.distance).take(getK)
        id -> closestNeighbors
      }
      .toDF(getIdentifierCol, getNeighborsCol)
      .select(
        col(getIdentifierCol).cast(identifierType).as(getIdentifierCol),
        col(getNeighborsCol).cast(ArrayType(StructType(Seq(
          StructField("neighbor", identifierType),
          StructField("distance", FloatType)
        )))).as(getNeighborsCol)
      )
  }

  override def transformSchema(schema: StructType): StructType = {
    val identifierType = schema(getIdentifierCol).dataType

    StructType(Seq(
      StructField(getIdentifierCol, identifierType),
      StructField(getNeighborsCol, ArrayType(StructType(Seq(StructField("neighbor", identifierType), StructField("distance", FloatType)))))
    ))
  }
}

abstract class KnnAlgorithm[TModel <: Model[TModel]](override val uid: String) extends Estimator[TModel] with KnnAlgorithmParams {

  import Udfs._

  def setIdentityCol(value: String): this.type = set(identifierCol, value)

  /** @group setParam */
  def setVectorCol(value: String): this.type = set(vectorCol, value)

  /** @group setParam */
  def setNeighborsCol(value: String): this.type = set(neighborsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  def setDistanceFunction(value: String): this.type = set(distanceFunction, value)

  /** @group setParam */
  def setExcludeSelf(value: Boolean): this.type = set(excludeSelf, value)

  override def fit(dataset: Dataset[_]): TModel = {

    import dataset.sparkSession.implicits._

    val vectorCol = dataset.schema(getVectorCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getVectorCol))
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getVectorCol))
      case _ => col(getVectorCol)
    }

    val partitioner = new PartitionIdPassthrough(getNumPartitions)

    val indicesRdd = dataset.select(col(getIdentifierCol).cast(StringType).as("id"),
      vectorCol.as("vector")).as[IndexItem]
      .map { item => (abs(item.id.hashCode) % getNumPartitions, item) }
      .rdd
      .partitionBy(partitioner)
      .mapPartitions( it =>
        if (it.hasNext) {

          val pairs = it.toList
          val partition = pairs.head._1
          val items = pairs.map(_._2)

          logInfo(s"Indexing ${items.size} items for partition $partition on host ${InetAddress.getLocalHost.getHostName}")

          val index = createIndex(items.size)
          index.addAll(items, progressUpdateInterval = 5000,  listener = (workDone, max) => logDebug(s"Indexed $workDone of $max items"))

          logInfo(s"Done indexing ${items.size} items for partition $partition on host ${InetAddress.getLocalHost.getHostName}")

          Iterator.single(partition -> index)
        } else Iterator.empty
        , preservesPartitioning = true)

    val model = createModel(uid, getNumPartitions, partitioner, indicesRdd)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    val identifierType = schema(getIdentifierCol).dataType

    StructType(Seq(
      StructField(getIdentifierCol, identifierType),
      StructField(getNeighborsCol, ArrayType(StructType(Seq(StructField("neighbor", identifierType), StructField("distance", FloatType)))))
    ))
  }

  override def copy(extra: ParamMap): Estimator[TModel] = defaultCopy(extra)

  /**
    * Create the index used to do the nearest neighbor search.
    *
    * @param maxItemCount maximum number of items the index can hold
    * @return create an index
    */
  protected def createIndex(maxItemCount: Int): Index[String, Array[Float], IndexItem, Float]

  /**
    * Creates the model to be returned from fitting the data.
    *
    * @param uid identifier
    * @param numPartitions how many partitions
    * @param partitioner the partitioner used to partition the data
    * @param indices rdd that holds the indices that are used to do the search
    * @return model
    */
  protected def createModel(uid: String,
                            numPartitions: Int,
                            partitioner: Partitioner,
                            indices: RDD[(Int, Index[String, Array[Float], IndexItem, Float])]): TModel

  protected def distanceFunctionByName(name: String): DistanceFunction[Array[Float], Float] = name match {
    case "bray-curtis" => floatBrayCurtisDistance
    case "canberra" => floatCanberraDistance
    case "correlation" => floatCorrelationDistance
    case "cosine" => floatCosineDistance
    case "euclidean" => floatEuclideanDistance
    case "inner-product" => floatInnerProduct
    case "manhattan" => floatManhattanDistance
    case _ => throw new IllegalArgumentException(s"$getDistanceFunction is not a valid distance function.")
  }

}

/**
  * Partitioner that uses precomputed partitions
  *
  * @param numPartitions number of partitions
  */
class PartitionIdPassthrough(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
}