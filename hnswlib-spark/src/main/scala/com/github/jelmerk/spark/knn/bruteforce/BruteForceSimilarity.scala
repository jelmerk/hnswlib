package com.github.jelmerk.spark.knn.bruteforce

import scala.reflect.runtime.universe._
import scala.reflect.ClassTag

import com.github.jelmerk.knn.scalalike.{DistanceFunction, Item}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.spark.knn._
import org.apache.spark.ml.Model
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}


/**
  * Companion class for BruteForceModel.
  */
object BruteForceSimilarityModel extends MLReadable[BruteForceSimilarityModel] {

  private[knn] class BruteForceModelReader extends KnnModelReader[BruteForceSimilarityModel] {

    override protected type TIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] = BruteForceIndex[TId, TVector, TItem, TDistance]

    override protected def createModel[
      TId: TypeTag,
      TVector: TypeTag,
      TItem <: Item[TId, TVector] with Product: TypeTag,
      TDistance : TypeTag
    ](uid: String, indices: RDD[(Int, BruteForceIndex[TId, TVector, TItem, TDistance])])
      (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance]) : BruteForceSimilarityModel =
        new GenericBruteForceModel[TId, TVector, TItem, TDistance](uid, indices)

  }

  override def read: MLReader[BruteForceSimilarityModel] = new BruteForceModelReader
}

/**
  * Model produced by a `BruteForceSimilarity`.
  */
abstract class BruteForceSimilarityModel extends Model[BruteForceSimilarityModel] with KnnModelParams with MLWritable


private[knn] class GenericBruteForceModel[
  TId : TypeTag,
  TVector : TypeTag,
  TItem <: Item[TId, TVector] with Product : TypeTag,
  TDistance : TypeTag
](override val uid: String, private[knn] val indices: RDD[(Int, BruteForceIndex[TId, TVector, TItem, TDistance])])
 (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance])
    extends BruteForceSimilarityModel with KnnModelSupport[BruteForceSimilarityModel, TId, TVector, TItem, TDistance, BruteForceIndex[TId, TVector, TItem, TDistance]] {

  override def transform(dataset: Dataset[_]): DataFrame = typedTransform(dataset)

  override def copy(extra: ParamMap): BruteForceSimilarityModel = {
    val copied = new GenericBruteForceModel[TId, TVector, TItem, TDistance](uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new KnnModelWriter[BruteForceSimilarityModel, TId, TVector, TItem, TDistance, BruteForceIndex[TId, TVector, TItem, TDistance]](this)
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

  /**
    * Create the index used to do the nearest neighbor search.
    *
    * @param dimensions   dimensionality of the items stored in the index
    * @param maxItemCount maximum number of items the index can hold
    * @return create an index
    */
  override protected def createIndex[TId, TVector, TItem <: Item[TId, TVector] with Product, TDistance]
      (dimensions: Int, maxItemCount: Int, distanceFunction: DistanceFunction[TVector, TDistance])(implicit distanceOrdering: Ordering[TDistance])
        : BruteForceIndex[TId, TVector, TItem, TDistance] =
            BruteForceIndex[TId, TVector, TItem, TDistance](
              dimensions,
              distanceFunction
            )

  override protected def createModel[
    TId: TypeTag,
    TVector: TypeTag,
    TItem <: Item[TId, TVector] with Product: TypeTag,
    TDistance : TypeTag
  ](uid: String, indices: RDD[(Int, BruteForceIndex[TId, TVector, TItem, TDistance])])
    (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance]) : BruteForceSimilarityModel =
      new GenericBruteForceModel[TId, TVector, TItem, TDistance](uid, indices)

}
