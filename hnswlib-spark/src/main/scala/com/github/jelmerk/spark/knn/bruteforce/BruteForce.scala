package com.github.jelmerk.spark.knn.bruteforce

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.spark.knn._
import org.apache.spark.rdd.RDD

/**
  * Companion class for BruteForceModel.
  */
object BruteForceModel extends MLReadable[BruteForceModel] {

  private[knn] class BruteForceModelReader extends KnnModelReader[BruteForceModel, BruteForceIndex[String, Array[Float], IndexItem, Float]] {
    override protected def createModel(uid: String,
                                       indices: RDD[(Int, (BruteForceIndex[String, Array[Float], IndexItem, Float], String, Array[Float]))]): BruteForceModel =
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
                      indices: RDD[(Int, (BruteForceIndex[String, Array[Float], IndexItem, Float], String, Array[Float]))])
  extends KnnModel[BruteForceModel, BruteForceIndex[String, Array[Float], IndexItem, Float]](uid, indices) with MLWritable {


  override def copy(extra: ParamMap): BruteForceModel = {
    val copied = new BruteForceModel(uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new KnnModelWriter[BruteForceModel, BruteForceIndex[String, Array[Float], IndexItem, Float]](this)
}

/**
  * Nearest neighbor search using a brute force approach. This will be very slow. It is in most cases not recommended
  * for production use. But can be used to determine the accuracy of an approximative index.
  *
  * @param uid identifier
  */
class BruteForce(override val uid: String) extends KnnAlgorithm[BruteForceModel, BruteForceIndex[String, Array[Float], IndexItem, Float]](uid)  {

  def this() = this(Identifiable.randomUID("brute_force"))

  override def createIndex(maxItemCount: Int): BruteForceIndex[String, Array[Float], IndexItem, Float] =
    BruteForceIndex[String, Array[Float], IndexItem, Float](distanceFunctionByName(getDistanceFunction))

  override def createModel(uid: String,
                           indices: RDD[(Int, (BruteForceIndex[String, Array[Float], IndexItem, Float], String, Array[Float]))]): BruteForceModel =
    new BruteForceModel(uid, indices)

}
