package com.github.jelmerk.spark.knn.bruteforce

import java.io.InputStream

import com.github.jelmerk.knn.ObjectSerializer

import scala.reflect.runtime.universe._
import scala.reflect.ClassTag
import com.github.jelmerk.knn.scalalike.{DistanceFunction, Item}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.spark.knn._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Companion class for BruteForceSimilarityModel.
  */
object BruteForceSimilarityModel extends MLReadable[BruteForceSimilarityModel] {

  private[knn] class BruteForceModelReader extends KnnModelReader[BruteForceSimilarityModel] {

    override protected def createModel[
      TId: TypeTag,
      TVector: TypeTag,
      TItem <: Item[TId, TVector] with Product: TypeTag,
      TDistance : TypeTag
    ](uid: String, outputDir: String, numPartitions: Int)
      (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance]) : BruteForceSimilarityModel =
        new BruteForceSimilarityModelImpl[TId, TVector, TItem, TDistance](uid, outputDir, numPartitions)

  }

  override def read: MLReader[BruteForceSimilarityModel] = new BruteForceModelReader
}

/**
  * Model produced by `BruteForceSimilarity`.
  */
abstract class BruteForceSimilarityModel extends KnnModelBase[BruteForceSimilarityModel] with KnnModelParams with MLWritable


private[knn] class BruteForceSimilarityModelImpl[
  TId : TypeTag,
  TVector : TypeTag,
  TItem <: Item[TId, TVector] with Product : TypeTag,
  TDistance : TypeTag
](override val uid: String, val outputDir: String, numPartitions: Int)
 (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance])
    extends BruteForceSimilarityModel with KnnModelOps[BruteForceSimilarityModel, TId, TVector, TItem, TDistance, BruteForceIndex[TId, TVector, TItem, TDistance]] {

  override def getNumPartitions: Int = numPartitions

  override def transform(dataset: Dataset[_]): DataFrame = typedTransform(dataset)

  override def copy(extra: ParamMap): BruteForceSimilarityModel = {
    val copied = new BruteForceSimilarityModelImpl[TId, TVector, TItem, TDistance](uid, outputDir, numPartitions)
    copyValues(copied, extra).setParent(parent)
  }

  override def transformSchema(schema: StructType): StructType = typedTransformSchema[TId](schema)

  override def write: MLWriter = new KnnModelWriter[BruteForceSimilarityModel, TId, TVector, TItem, TDistance, BruteForceIndex[TId, TVector, TItem, TDistance]](this)

  override protected def loadIndex(in: InputStream): BruteForceIndex[TId, TVector, TItem, TDistance] =
    BruteForceIndex.loadFromInputStream[TId, TVector, TItem, TDistance](in)

}

/**
  * Nearest neighbor search using a brute force approach. This will be very slow. It is in most cases not recommended
  * for production use. But can be used to determine the accuracy of an approximative index.
  *
  * @param uid identifier
  */
class BruteForceSimilarity(override val uid: String) extends KnnAlgorithm[BruteForceSimilarityModel](uid) {

  override protected type TIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] = BruteForceIndex[TId, TVector, TItem, TDistance]

  def this() = this(Identifiable.randomUID("brute_force"))

  override protected def createIndex[TId, TVector, TItem <: Item[TId, TVector] with Product, TDistance]
    (dimensions: Int, maxItemCount: Int, distanceFunction: DistanceFunction[TVector, TDistance])(implicit distanceOrdering: Ordering[TDistance], idSerializer: ObjectSerializer[TId], itemSerializer: ObjectSerializer[TItem])
        : BruteForceIndex[TId, TVector, TItem, TDistance] =
            BruteForceIndex[TId, TVector, TItem, TDistance](
              dimensions,
              distanceFunction
            )

  override protected def loadIndex[TId, TVector, TItem <: Item[TId, TVector] with Product, TDistance]
    (inputStream: InputStream, minCapacity: Int): BruteForceIndex[TId, TVector, TItem, TDistance] = BruteForceIndex.loadFromInputStream(inputStream)

  override protected def createModel[
    TId: TypeTag,
    TVector: TypeTag,
    TItem <: Item[TId, TVector] with Product: TypeTag,
    TDistance : TypeTag
  ](uid: String, outputDir: String, numPartitions: Int)
    (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance]) : BruteForceSimilarityModel =
      new BruteForceSimilarityModelImpl[TId, TVector, TItem, TDistance](uid, outputDir, numPartitions)

}
