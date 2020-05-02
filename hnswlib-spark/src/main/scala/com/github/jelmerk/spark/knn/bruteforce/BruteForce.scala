package com.github.jelmerk.spark.knn.bruteforce

import com.github.jelmerk.knn.scalalike.{DistanceFunction, Item}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.spark.knn._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Companion class for BruteForceModel.
  */
object BruteForceModel extends MLReadable[BruteForceModel] {

  private[knn] class BruteForceModelReader extends KnnModelReader[BruteForceModel] {

    override protected type IndexType[TId, TVector, TItem <: Item[TId, TVector], TDistance] =
      BruteForceIndex[TId, TVector, TItem, TDistance]

    override protected def createModel(uid: String, indices: Either[RDD[(Int, (BruteForceIndex[String, Array[Float], VectorIndexItemDense, Float], String, Array[Float]))],
                                                                    RDD[(Int, (BruteForceIndex[String, Vector, VectorIndexItemSparse, Float], String, Vector))]]): BruteForceModel =
      new BruteForceModel(uid, indices)
  }

  override def read: MLReader[BruteForceModel] = new BruteForceModelReader
}

/**
  * Model produced by a `BruteForce`.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  */
class BruteForceModel private[bruteforce](override val uid: String,
                                          override val indices: Either[RDD[(Int, (BruteForceIndex[String, Array[Float], VectorIndexItemDense, Float], String, Array[Float]))],
                                                                       RDD[(Int, (BruteForceIndex[String, Vector, VectorIndexItemSparse, Float], String, Vector))]])
  extends KnnModel[BruteForceModel](uid) with MLWritable {

  override protected type IndexType[TId, TVector, TItem <: Item[TId, TVector], TDistance] =
    BruteForceIndex[TId, TVector, TItem, TDistance]

  override def copy(extra: ParamMap): BruteForceModel = {
    val copied = new BruteForceModel(uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new KnnModelWriter[BruteForceModel](this)
}

/**
  * Nearest neighbor search using a brute force approach. This will be very slow. It is in most cases not recommended
  * for production use. But can be used to determine the accuracy of an approximative index.
  *
  * @param uid identifier
  */
class BruteForce(override val uid: String) extends KnnAlgorithm[BruteForceModel](uid)  {

  def this() = this(Identifiable.randomUID("brute_force"))

  override protected type IndexType[TId, TVector, TItem <: Item[TId, TVector], TDistance] =
    BruteForceIndex[TId, TVector, TItem, TDistance]

  override protected def createIndex[TVector, TItem <: Item[String, TVector] with Product](dimensions: Int,
                                                                                           maxItemCount: Int,
                                                                                           distanceFunction: DistanceFunction[TVector, Float]): BruteForceIndex[String, TVector, TItem, Float] =
    BruteForceIndex[String, TVector, TItem, Float](dimensions, distanceFunction)

  override protected def createModel(uid: String, indices: Either[RDD[(Int, (BruteForceIndex[String, Array[Float], VectorIndexItemDense, Float], String, Array[Float]))],
                                                                  RDD[(Int, (BruteForceIndex[String, Vector, VectorIndexItemSparse, Float], String, Vector))]]): BruteForceModel =
    new BruteForceModel(uid, indices)


}
