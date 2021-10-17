package com.github.jelmerk.spark.partitioning.reldg

import com.github.jelmerk.knn.util.ReLdg.{Node, SimpleNode}
import com.github.jelmerk.knn.scalalike.hnsw.HnswIndex
import com.github.jelmerk.knn.scalalike.{DistanceFunction, Item, doubleBrayCurtisDistance, doubleCanberraDistance, doubleCorrelationDistance, doubleCosineDistance, doubleEuclideanDistance, doubleInnerProduct, doubleManhattanDistance, floatBrayCurtisDistance, floatCanberraDistance, floatCorrelationDistance, floatCosineDistance, floatEuclideanDistance, floatInnerProduct, floatManhattanDistance}
import com.github.jelmerk.knn.util.ReLdg
import com.github.jelmerk.spark.linalg.functions.VectorDistanceFunctions
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DoubleType, FloatType, IntegerType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConverters._
import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import scala.util.Try


private[reldg] trait ReldgParams extends ReldgModelParams  {

  /**
   * Param for the distance function to use. One of "bray-curtis", "canberra",  "cosine", "correlation", "euclidean",
   * "inner-product", "manhattan" or the fully qualified classname of a distance function
   * Default: "cosine"
   *
   * @group param
   */
  final val distanceFunction = new Param[String](this, "distanceFunction", "distance function to use")

  /** @group getParam */
  final def getDistanceFunction: String = $(distanceFunction)

  setDefault(distanceFunction -> "cosine")

}

private[reldg] trait ReldgModelParams extends Params with HasFeaturesCol {

  final val partitionCol: Param[String] = new Param[String](this, "partitionCol", "primary partition")

  final def getPartitionCol: String = $(partitionCol)

  final val queryPartitionsCol: Param[String] = new Param[String](this, "queryPartitionsCol", "query partitions")

  final def getQueryPartitionsCol: String = $(queryPartitionsCol)

  final val k = new IntParam(this, "k", "The number of required neighbors." +
    "Must be > 1.", ParamValidators.gt(1))

  final def getK: Int = $(k)

  final val numPartitions = new IntParam(this, "numPartitions", "The number of partitions to create." +
    "Must be > 1.", ParamValidators.gt(1))

  final def getNumPartitions: Int = $(numPartitions)

  final val numIterations = new IntParam(this, "numIterations", "The number of iterations." +
    "Must be > 1.", ParamValidators.gt(1))

  final def getNumIterations: Int = $(numIterations)

  final val epsilon = new DoubleParam(this, "epsilon", "The epsilon.")

  final def getEpsilon: Double = $(epsilon)

  setDefault(numIterations -> 10, epsilon -> 0.0, partitionCol -> "partition", k -> 10,
             queryPartitionsCol -> "query_partitions")

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val partitionField = StructField(getPartitionCol, IntegerType)
    val queryPartitionsField = StructField(getQueryPartitionsCol, ArrayType(IntegerType))
    StructType(schema.fields :+ partitionField :+ queryPartitionsField)
  }
}

object ReldgModel extends MLReadable[ReldgModel] {

  override def read: MLReader[ReldgModel] = new ReldgModelReader

  private class ReldgModelReader extends MLReader[ReldgModel] {
    override def load(path: String): ReldgModel = ???
  }

  private[reldg] class ReldgModelWriter(instance: ReldgModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = ???

  }
}


abstract class ReldgModel extends Model[ReldgModel] with ReldgModelParams with MLWritable {

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setPartitionCol(value: String): this.type = set(partitionCol, value)

  /** @group setParam */
  def setQueryPartitionsCol(value: String): this.type = set(queryPartitionsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  def setNumIterations(value: Int): this.type = set(numIterations, value)

  /** @group setParam */
  def seEpsilon(value: Double): this.type = set(epsilon, value)
}

class ReldgModelImpl[
  TVector : TypeTag,
  TItem <: Item[Int, TVector] with Product : TypeTag,
  TDistance : TypeTag
](override val uid: String, index: HnswIndex[Int, TVector, TItem, TDistance]) extends ReldgModel {

  override def copy(extra:  ParamMap): ReldgModel = {
    val copied = new ReldgModelImpl[TVector, TItem, TDistance](uid, index)
    copyValues(copied, extra).setParent(parent)
  }
  override def transform(dataset: Dataset[_]): DataFrame = {

    import dataset.sparkSession.implicits._

    val broadcastedIndex = dataset.sparkSession.sparkContext.broadcast(index)

    val findNearest: UserDefinedFunction = udf { value: TVector =>
      val nearest = broadcastedIndex.value.findNearest(value, getK)
      nearest.map(_.item.id)
    }

    val ds = dataset
      .withColumn("_nearest_clusters", findNearest(col(getFeaturesCol)))
      .persist(StorageLevel.MEMORY_AND_DISK)

    val clusterCounts: Map[Int, Long] = ds
      .groupBy($"_nearest_clusters"(0))
      .count()
      .as[(Int, Long)]
      .collect()
      .toMap

    println("cluster counts :")
    clusterCounts.foreach { case (cluster, count) =>
      println("  " + cluster + " " + count)
    }

    val nodes = index
      .map { item =>
        val clusterId = item.id
        val weight = clusterCounts.getOrElse(clusterId, 0L) // TODO how can this ever not exist ?? if its a centroid surely at least one element should be in it
        val edges = index.connections(clusterId, level = 0).map(_.id).toArray
        new SimpleNode(clusterId, edges, weight.toInt) : Node
      }
      .toArray


    println("num partitions " + getNumPartitions)

    val reordered = ReLdg.bfsDisconnected(nodes)
    val partitionAssignments = ReLdg.reldg(reordered, getNumPartitions, getNumIterations, getEpsilon)

    println("partitionAssignments " + partitionAssignments.size)


    val toPartition: UserDefinedFunction = udf { clusterId: Int =>
      partitionAssignments(clusterId)
    }

    val toPartitions: UserDefinedFunction = udf { clusterIds: Seq[Int] =>
      clusterIds.map(partitionAssignments.apply).distinct
    }

    ds
      .withColumn(getPartitionCol, toPartition($"_nearest_clusters"(0)))
      .withColumn(getQueryPartitionsCol, toPartitions($"_nearest_clusters"))
      .drop("_nearest_clusters")

  }

  override def transformSchema(schema:  StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new ReldgModel.ReldgModelWriter(this)
}




/**
 * Balanced partitioning by using the REstreamed Linear Deterministic Greedy algorithm on the bottom layer of
 * a HNSW graph. This is not a scalable algorithm
 *
 * @param uid identifier
 */
class Reldg(override val uid: String) extends Estimator[ReldgModel] with ReldgParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("reldg"))

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setPartitionCol(value: String): this.type = set(partitionCol, value)

  /** @group setParam */
  def setQueryPartitionsCol(value: String): this.type = set(queryPartitionsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  def setEpsilon(value: Double): this.type = set(epsilon, value)

  /** @group setParam */
  def setDistanceFunction(value: String): this.type = set(distanceFunction, value)

  override def fit(dataset: Dataset[_]): ReldgModel = {

    val vectorType = dataset.schema(getFeaturesCol).dataType

    val model = vectorType match {
      case ArrayType(FloatType, _) => typedFit[Array[Float], IntFloatArrayIndexItem, Float](dataset)
      case ArrayType(DoubleType, _) => typedFit[Array[Double], IntDoubleArrayIndexItem, Double](dataset)
      case t if t.typeName == "vector" => typedFit[Vector, IntVectorIndexItem, Double](dataset)
      case _ => throw new IllegalArgumentException(s"Cannot Supported vectors are array<float>, array<double> and vector ")
    }

    copyValues(model)
  }

  private def typedFit[
    TVector : TypeTag,
    TItem <: Item[Int, TVector] with Product : TypeTag,
    TDistance: TypeTag
  ](dataset: Dataset[_])
    (implicit  evVector: ClassTag[TVector], evItem: ClassTag[TItem], distanceNumeric: Numeric[TDistance],
     distanceFunctionFactory: String => DistanceFunction[TVector, TDistance])
  : ReldgModel = {

    import dataset.sparkSession.implicits._

    val window = Window.orderBy($"vector")
    val items = dataset.
      select(
        col(getFeaturesCol).as("vector")
      )
      .withColumn("id", row_number().over(window) - 1)
      .as[TItem]
      .collect()

    val dimensions = items.head.dimensions()

    val distanceFunction = distanceFunctionFactory(getDistanceFunction)

    val index: HnswIndex[Int, TVector, TItem, TDistance] = HnswIndex[Int, TVector, TItem, TDistance](
      dimensions,
      distanceFunction,
      items.length
    )

    items.foreach(index.add)

    new ReldgModelImpl(uid, index)

  }

  override def copy(extra: ParamMap): Estimator[ReldgModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema


  // TODO: all of this stuff is duplicated
  implicit private def floatArrayDistanceFunction(name: String): DistanceFunction[Array[Float], Float] = name match {
    case "bray-curtis" => floatBrayCurtisDistance
    case "canberra" => floatCanberraDistance
    case "correlation" => floatCorrelationDistance
    case "cosine" => floatCosineDistance
    case "euclidean" => floatEuclideanDistance
    case "inner-product" => floatInnerProduct
    case "manhattan" => floatManhattanDistance
    case value => userDistanceFunction(value)
  }

  implicit private def doubleArrayDistanceFunction(name: String): DistanceFunction[Array[Double], Double] = name match {
    case "bray-curtis" => doubleBrayCurtisDistance
    case "canberra" => doubleCanberraDistance
    case "correlation" => doubleCorrelationDistance
    case "cosine" => doubleCosineDistance
    case "euclidean" => doubleEuclideanDistance
    case "inner-product" => doubleInnerProduct
    case "manhattan" => doubleManhattanDistance
    case value => userDistanceFunction(value)
  }

  implicit private def vectorDistanceFunction(name: String): DistanceFunction[Vector, Double] = name match {
    case "bray-curtis" => VectorDistanceFunctions.brayCurtisDistance
    case "canberra" => VectorDistanceFunctions.canberraDistance
    case "correlation" => VectorDistanceFunctions.correlationDistance
    case "cosine" => VectorDistanceFunctions.cosineDistance
    case "euclidean" => VectorDistanceFunctions.euclideanDistance
    case "inner-product" => VectorDistanceFunctions.innerProduct
    case "manhattan" => VectorDistanceFunctions.manhattanDistance
    case value => userDistanceFunction(value)
  }

  private def userDistanceFunction[TVector, TDistance](name: String): DistanceFunction[TVector, TDistance] =
    Try(Class.forName(name).getDeclaredConstructor().newInstance())
      .toOption
      .collect { case f: DistanceFunction[TVector @unchecked, TDistance @unchecked] => f }
      .getOrElse(throw new IllegalArgumentException(s"$name is not a valid distance functions."))
}

private[reldg] case class IntFloatArrayIndexItem(id: Int, vector: Array[Float]) extends Item[Int, Array[Float]] {
  override def dimensions: Int = vector.length
}

private[reldg] case class IntVectorIndexItem(id: Int, vector: Vector) extends Item[Int, Vector] {
  override def dimensions: Int = vector.size
}

private[reldg] case class IntDoubleArrayIndexItem(id: Int, vector: Array[Double]) extends Item[Int, Array[Double]] {
  override def dimensions: Int = vector.length
}