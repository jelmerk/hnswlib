package com.github.jelmerk.knn.spark.ml.hnsw

import java.net.InetAddress

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.hnsw._
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.storage.StorageLevel

case class PartitionAndIndexItem(partition: Int, item: IndexItem)

case class IndexItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]]

case class Neighbor(id: String, distance: Float)

object Udfs {

  val vectorToFloatArray: UserDefinedFunction = udf { vector: Vector => vector.toArray.map(_.toFloat) }

  val doubleArrayToFloatArray: UserDefinedFunction = udf { vector: Array[Double] => vector.map(_.toFloat) }
}

trait HnswModelParams extends Params {

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
    * Param for the distance function to use. One of "cosine", "inner-product"
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

  /** @group getParam */
  def getK: Int = $(k)

}

trait HnswParams extends HnswModelParams {

  /**
    * The number of bi-directional links created for every new element during construction.
    *
    * Default: 16
    *
    * @group param
    */
  val m = new IntParam(this, "m",
    "number of bi-directional links created for every new element during construction", ParamValidators.gt(0))

  /** @group getParam */
  def getM: Int = $(m)

  /**
    * Size of the dynamic list for the nearest neighbors (used during the search).
    * Default: 10
    *
    * @group param
    */
  val ef = new IntParam(this, "ef",
    "size of the dynamic list for the nearest neighbors (used during the search)", ParamValidators.gt(0))

  /** @group getParam */
  def getEf: Int = $(ef)

  /**
    * Has the same meaning as ef, but controls the index time / index precision.
    * Default: 200
    *
    * @group param
    */
  val efConstruction = new IntParam(this, "efConstruction",
    "has the same meaning as ef, but controls the index time / index precision", ParamValidators.gt(0))

  /** @group getParam */
  def getEfConstruction: Int = $(efConstruction)

  /**
    * Number of partitions (default: 1)
    */
  val numPartitions = new IntParam(this, "numPartitions",
    "number of partitions", ParamValidators.gt(0))

  /** @group getParam */
  def getNumPartitions: Int = $(numPartitions)


  setDefault(m -> 16, ef -> 10, efConstruction -> 200, numPartitions -> 1, k -> 5,
    neighborsCol -> "neighbors", identifierCol -> "id", vectorCol -> "vector", distanceFunction -> "cosine")
}

object CustomEncoders {

  implicit val indexEncoder: Encoder[HnswIndex[String, Array[Float], IndexItem, Float]] =
    Encoders.javaSerialization[HnswIndex[String, Array[Float], IndexItem, Float]]

}

class HnswModel(override val uid: String,
                numPartitions: Int,
                partitioner: Partitioner,
                indices: RDD[(Int, HnswIndex[String, Array[Float], IndexItem, Float])])

  extends Model[HnswModel] with HnswModelParams {

  import Udfs._

  /** @group setParam */
  def setNeighborsCol(value: String): this.type = set(neighborsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  override def copy(extra: ParamMap): HnswModel = {
    val copied = new HnswModel(uid, numPartitions, partitioner, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._
    import CustomEncoders._

    val identifierType = dataset.schema(getIdentifierCol).dataType

    implicit val partitionAndIndexEncoder: Encoder[(Int, String, Array[Float], HnswIndex[String, Array[Float], IndexItem, Float])] =
      Encoders.tuple(Encoders.scalaInt, Encoders.STRING, ExpressionEncoder[Array[Float]](), indexEncoder)

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
      .cogroup(indices).flatMap { case (_, (itemsIter, indicesIter)) =>
        indicesIter.headOption.map { index =>
          itemsIter.map { case (id, vector) =>
            val neighbors = index.findNearest(vector, getK + 1)
              .collect { case SearchResult(item, distance) if item.id != id => Neighbor(item.id, distance) }
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
          StructField("id", identifierType),
          StructField("distance", FloatType)
        )))).as(getNeighborsCol)
      )
  }

  override def transformSchema(schema: StructType): StructType = {
    val identifierType = schema(getIdentifierCol).dataType

    StructType(Seq(
      StructField(getIdentifierCol, identifierType),
      StructField(getNeighborsCol, ArrayType(StructType(Seq(StructField("id", identifierType), StructField("distance", FloatType)))))
    ))
  }
}

class Hnsw(override val uid: String) extends Estimator[HnswModel] with HnswParams {

  import CustomEncoders._
  import Udfs._

  def this() = this(Identifiable.randomUID("hnsw"))

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
  def setM(value: Int): this.type = set(m, value)

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  /** @group setParam */
  def setEfConstruction(value: Int): this.type = set(efConstruction, value)

  /** @group setParam */
  def setDistanceFunction(value: String): this.type = set(distanceFunction, value)

  override def fit(dataset: Dataset[_]): HnswModel = {
    // TODO i think fit should do nothing

    import dataset.sparkSession.implicits._

    implicit val partitionAndIndexEncoder: Encoder[(Int, HnswIndex[String, Array[Float], IndexItem, Float])] =
      Encoders.tuple(Encoders.scalaInt, indexEncoder)

    val vectorCol = dataset.schema(getVectorCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getVectorCol))
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getVectorCol))
      case _ => col(getVectorCol)
    }

    val partitioner = new PartitionIdPassthrough(getNumPartitions)

    val indicesRdd = dataset.select(col(getIdentifierCol).cast(StringType).as("id"),
                                 vectorCol.as("vector")).as[IndexItem]
      .map { item => (item.id.hashCode % getNumPartitions, item) }
      .rdd
      .partitionBy(partitioner)
      .mapPartitions( it =>
        if (it.hasNext) {

          val pairs = it.toList
          val partition = pairs.head._1
          val items = pairs.map(_._2)

          logInfo(s"Indexing ${pairs.size} items for partition $partition on host ${InetAddress.getLocalHost.getHostName}")

          val index = HnswIndex[String, Array[Float], IndexItem, Float](
            distanceFunction = distanceFunctionByName(getDistanceFunction),
            maxItemCount = pairs.size,
            m = getM,
            ef = getEf,
            efConstruction = getEfConstruction
          )

          index.addAll(items, progressUpdateInterval = 100,  listener = (workDone, max) => logInfo(s"Indexed $workDone of $max items"))

          logInfo(s"Done indexing ${pairs.size} items for partition $partition on host ${InetAddress.getLocalHost.getHostName}")

          Iterator.single(partition -> index)
        } else Iterator.empty
      , preservesPartitioning = true)

    val model = new HnswModel(uid, getNumPartitions, partitioner, indicesRdd).setParent(this)
    copyValues(model)
  }

  override def copy(extra: ParamMap): Estimator[HnswModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val identifierType = schema(getIdentifierCol).dataType

    StructType(Seq(
      StructField(getIdentifierCol, identifierType),
      StructField(getNeighborsCol, ArrayType(StructType(Seq(StructField("id", identifierType), StructField("distance", FloatType)))))
    ))
  }

  private def distanceFunctionByName(name: String): DistanceFunction[Array[Float], Float] = name match {
    case "cosine" => floatCosineDistance
    case "inner-product" => floatInnerProduct
    case _ => throw new IllegalArgumentException(s"$getDistanceFunction is not a valid distance function.")
  }
}

class PartitionIdPassthrough(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
}
