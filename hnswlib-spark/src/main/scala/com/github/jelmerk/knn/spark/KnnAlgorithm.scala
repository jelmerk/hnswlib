package com.github.jelmerk.knn.spark

import java.net.InetAddress

import scala.util.Try

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
private[spark] case class IndexItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]]

/**
  * Neighbor of an item
  *
  * @param neighbor identifies the neighbor
  * @param distance distance to the item
  */
private[spark] case class Neighbor(neighbor: String, distance: Float) extends Comparable[Neighbor] {
  override def compareTo(other: Neighbor): Int = distance.compareTo(other.distance)
}


private[spark] object Udfs {

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
    * Param for the distance function to use. One of "bray-curtis", "canberra",  "cosine", "correlation", "euclidean",
    * "inner-product", "manhattan" or the fully qualified classname of a distance function
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
    * Param that indicates whether to include the row identifier as a candidate neighbor
    * Default: false
    *
    * @group param
    */
  val excludeSelf = new BooleanParam(this, "excludeSelf", "whether or not to exclude the query row_id")

  /**
    * Whether to include the row identifier as a candidate neighbor.
    *
    * @group getParam
    * */
  def getExcludeSelf: Boolean = $(excludeSelf)

  setDefault(k -> 5, neighborsCol -> "neighbors", identifierCol -> "id", vectorCol -> "vector",
    distanceFunction -> "cosine", excludeSelf -> false)

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
  * @param indices rdd that holds the indices that are used to do the search
  * @tparam TModel model type
  */
abstract class KnnModel[TModel <: Model[TModel]](override val uid: String,
                                                 indices: RDD[(Int, (Index[String, Array[Float], IndexItem, Float], String, Array[Float]))])
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

    // select the item vector from the query dataframe, transform vectors or double arrays into float arrays
    // for performance reasons

    val vectorCol = dataset.schema(getVectorCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getVectorCol)) // VectorUDT is not accessible
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getVectorCol))
      case _ => col(getVectorCol)
    }

    // duplicate the rows in the query dataset with the number of partitions, assign a different partition to each copy

    val queryRdd = dataset
      .select(
        col(getIdentifierCol).cast(StringType),
        vectorCol.as(getVectorCol)
      )
      .withColumn("partition", explode(array(0 until indices.getNumPartitions map lit: _*)))
      .as[(String, Array[Float], Int)]
      .rdd
      .map { case (id, vector, partition) => (partition, (null.asInstanceOf[Index[String, Array[Float], IndexItem, Float]], id, vector)) }
      .partitionBy(indices.partitioner.get)

    // combine the indices rdd and query rdds into a single rdd and make sure the first row of the unioned rdd is our index

    val unioned = indices
      .union(queryRdd)

    // map over all the rows in the partition, hold on on to the index stored in the first row and
    // use it to find the nearest neighbors of the remaining rows

    val neighborsOnAllShards = unioned.mapPartitions { it =>
      if (it.hasNext) {
        val (partition, (index, _, _)) = it.next()

        if (index == null) {
          logInfo(f"partition $partition%04d: No index on partition, not querying anything.")
          Iterator.empty
        } else {
          new LoggingIterator(partition,
            it.grouped(20480).flatMap { grouped =>

              // use scala's parallel collections to speed up querying

              grouped.par.map { case (_, (_, id, vector)) =>

                val fetchSize =
                  if (getExcludeSelf) getK + 1
                  else getK

                val neighbors = index.findNearest(vector, fetchSize)
                  .collect { case SearchResult(item, distance)
                    if !getExcludeSelf || item.id != id => Neighbor(item.id, distance) }

                val queue = new BoundedPriorityQueue[Neighbor](getK)
                queue ++= neighbors

                id -> queue
              }
            }
          )
        }
      } else Iterator.empty
    }

    // reduce the top k neighbors on each shard to the top k neighbors over all shards, holding on to only the best matches

    val topNeighbors = neighborsOnAllShards
      .reduceByKey { case (neighborsA, neighborsB) =>
        neighborsA ++= neighborsB
        neighborsA
      }
      .mapValues(_.toArray.sorted)

    // transform the rdd into our output dataframe

    topNeighbors
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

  private class LoggingIterator[T](partition: Int, delegate: Iterator[T]) extends Iterator[T] {

    private[this] var count = 0
    private[this] var first = true

    override def hasNext: Boolean = delegate.hasNext

    override def next(): T = {
      if (first) {
        logInfo(f"partition $partition%04d: started querying on host ${InetAddress.getLocalHost.getHostName}")
        first  = false
      }

      val value = delegate.next()

      count += 1

      if (!hasNext) {
        logInfo(f"partition $partition%04d: finished querying $count items on host ${InetAddress.getLocalHost.getHostName}")
      }

      value
    }
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

    // read the id and vector from the input dataset and and repartition them over numPartitions amount of partitions.
    // Transform vectors or double arrays into float arrays for performance reasons.

    val partitionedIndexItems = dataset
      .select(
        col(getIdentifierCol).cast(StringType).as("id"),
        vectorCol.as("vector")
      ).as[IndexItem]
      .map { item => (abs(item.id.hashCode) % getNumPartitions, item) }
      .rdd
      .partitionBy(partitioner)

    // On each partition collect all the items into memory and construct the HNSW indices.
    // The result is a rdd that has a single row per partition containing the index

    val indicesRdd = partitionedIndexItems
      .mapPartitionsWithIndex((partition, it) =>
        if (it.hasNext) {
          val items = it.map{ case (_, indexItem) => indexItem}.toList

          logInfo(f"partition $partition%04d: indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          val index = createIndex(items.size)
          index.addAll(items, progressUpdateInterval = 5000, listener = (workDone, max) => logDebug(f"partition $partition%04d: Indexed $workDone of $max items"))

          logInfo(f"partition $partition%04d: done indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          Iterator.single(partition -> Tuple3(index, null.asInstanceOf[String], null.asInstanceOf[Array[Float]]))
        } else Iterator.empty
        , preservesPartitioning = true)

    val model = createModel(uid, indicesRdd)
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
    * @param indices rdd that holds the indices that are used to do the search
    * @return model
    */
  protected def createModel(uid: String,
                            indices: RDD[(Int, (Index[String, Array[Float], IndexItem, Float], String, Array[Float]))]): TModel

  protected def distanceFunctionByName(name: String): DistanceFunction[Array[Float], Float] = name match {
    case "bray-curtis" => floatBrayCurtisDistance
    case "canberra" => floatCanberraDistance
    case "correlation" => floatCorrelationDistance
    case "cosine" => floatCosineDistance
    case "euclidean" => floatEuclideanDistance
    case "inner-product" => floatInnerProduct
    case "manhattan" => floatManhattanDistance
    case value =>
      Try(Class.forName(value).getDeclaredConstructor().newInstance())
        .toOption
        .collect { case f: DistanceFunction[Array[Float] @unchecked, Float @unchecked] => f }
        .getOrElse(throw new IllegalArgumentException(s"$value is not a valid distance function."))
  }

}

/**
  * Partitioner that uses precomputed partitions
  *
  * @param numPartitions number of partitions
  */
private[spark] class PartitionIdPassthrough(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
}