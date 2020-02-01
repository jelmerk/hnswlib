package com.github.jelmerk.spark.knn.bruteforce

import com.github.jelmerk.knn.scalalike.SparseVector
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.spark.knn._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD

/**
  * Companion class for SBruteForceModel.
  */
object SBruteForceModel extends MLReadable[SBruteForceModel] {

  private[knn] class SBruteForceModelReader extends KnnModelReader[
    SBruteForceModel,
    String,
    SparseVector[Array[Float]],
    SparseVectorIndexItem,
    BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]] {
    override protected def createModel(uid: String,
                                       indices: RDD[(Int, (BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float], String, SparseVector[Array[Float]]))]): SBruteForceModel =
      new SBruteForceModel(uid, indices)
  }

  override def read: MLReader[SBruteForceModel] = new SBruteForceModelReader
}

/**
  * Model produced by a `SBruteForce`.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  */
class SBruteForceModel private[bruteforce](override val uid: String,
                                           indices: RDD[(Int, (BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float], String, SparseVector[Array[Float]]))])
  extends SparseVectorKnnModel[SBruteForceModel, BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]](uid, indices)
    with MLWritable {

  override def copy(extra: ParamMap): SBruteForceModel = {
    val copied = new SBruteForceModel(uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new KnnModelWriter[SBruteForceModel, String, SparseVector[Array[Float]], SparseVectorIndexItem, BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]](this)
}

/**
  * Nearest neighbor search using a brute force approach. This will be very slow. It is in most cases not recommended
  * for production use. But can be used to determine the accuracy of an approximative index.
  *
  * @param uid identifier
  */
class SBruteForce(override val uid: String)
  extends SparseVectorKnnAlgorithmBase[SBruteForceModel, BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]](uid) {

  def this() = this(Identifiable.randomUID("brute_force"))

  override def createIndex(maxItemCount: Int): BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float] =
    BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float](distanceFunctionByName(getDistanceFunction))

  override def createModel(uid: String,
                           indices: RDD[(Int, (BruteForceIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float], String, SparseVector[Array[Float]]))]): SBruteForceModel =
    new SBruteForceModel(uid, indices)

}
