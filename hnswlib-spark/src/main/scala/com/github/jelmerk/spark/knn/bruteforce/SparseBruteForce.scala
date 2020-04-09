package com.github.jelmerk.spark.knn.bruteforce

import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.spark.knn._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD

/**
  * Companion class for SparseBruteForceModel.
  */
object SparseSparseBruteForceModel extends MLReadable[SparseBruteForceModel] {

  private[knn] class SparseBruteForceModelReader extends KnnModelReader[SparseBruteForceModel, String, Vector, SparseVectorIndexItem, BruteForceIndex[String, Vector, SparseVectorIndexItem, Float]] {
    override protected def createModel(uid: String,
                                       indices: RDD[(Int, (BruteForceIndex[String, Vector, SparseVectorIndexItem, Float], String, Vector))]): SparseBruteForceModel =
      new SparseBruteForceModel(uid, indices)
  }

  override def read: MLReader[SparseBruteForceModel] = new SparseBruteForceModelReader
}

/**
  * Model produced by a `SparseBruteForce`.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  */
class SparseBruteForceModel private[bruteforce](override val uid: String,
                                                indices: RDD[(Int, (BruteForceIndex[String, Vector, SparseVectorIndexItem, Float], String, Vector))])
  extends SparseVectorKnnModel[SparseBruteForceModel, BruteForceIndex[String, Vector, SparseVectorIndexItem, Float]](uid, indices) with MLWritable {

  override def copy(extra: ParamMap): SparseBruteForceModel = {
    val copied = new SparseBruteForceModel(uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new KnnModelWriter[SparseBruteForceModel, String, Vector, SparseVectorIndexItem, BruteForceIndex[String, Vector, SparseVectorIndexItem, Float]](this)
}

/**
  * Nearest neighbor search using a brute force approach. This will be very slow. It is in most cases not recommended
  * for production use. But can be used to determine the accuracy of an approximative index.
  *
  * @param uid identifier
  */
class SparseBruteForce(override val uid: String) extends SparseVectorKnnAlgorithm[SparseBruteForceModel, BruteForceIndex[String, Vector, SparseVectorIndexItem, Float]](uid)  {

  def this() = this(Identifiable.randomUID("brute_force"))

  override def createIndex(dimensions: Int, maxItemCount: Int): BruteForceIndex[String, Vector, SparseVectorIndexItem, Float] =
    BruteForceIndex[String, Vector, SparseVectorIndexItem, Float](dimensions, distanceFunctionByName(getDistanceFunction))

  override def createModel(uid: String,
                           indices: RDD[(Int, (BruteForceIndex[String, Vector, SparseVectorIndexItem, Float], String, Vector))]): SparseBruteForceModel =
    new SparseBruteForceModel(uid, indices)

}
