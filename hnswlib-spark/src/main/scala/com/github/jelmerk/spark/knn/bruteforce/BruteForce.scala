package com.github.jelmerk.spark.knn.bruteforce

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.spark.knn._
import org.apache.spark.rdd.RDD

/**
  * Model produced by a `BruteForce`.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  */
class BruteForceModel(override val uid: String,
                      centroidsIndexOption: Option[Index[Int, Array[Float], CentroidIndexItem, Float]],
                      indices: RDD[(Int, (Index[String, Array[Float], IndexItem, Float], String, Array[Float]))])
  extends KnnModel[BruteForceModel](uid, centroidsIndexOption, indices) {

  override def copy(extra: ParamMap): BruteForceModel = {
    val copied = new BruteForceModel(uid, centroidsIndexOption, indices)
    copyValues(copied, extra).setParent(parent)
  }

}

/**
  * Nearest neighbor search using a brute force approach. This will be very slow. It is in most cases not recommended
  * for production use. But can be used to determine the accuracy of an approximative index.
  *
  * @param uid identifier
  */
class BruteForce(override val uid: String) extends KnnAlgorithm[BruteForceModel](uid)  {

  def this() = this(Identifiable.randomUID("brute_force"))

  override def createIndex(maxItemCount: Int): Index[String, Array[Float], IndexItem, Float] =
    BruteForceIndex[String, Array[Float], IndexItem, Float](distanceFunctionByName(getDistanceFunction))

  override def createModel(uid: String,
                           centroidsIndexOption: Option[Index[Int, Array[Float], CentroidIndexItem, Float]],
                           indices: RDD[(Int, (Index[String, Array[Float], IndexItem, Float], String, Array[Float]))]): BruteForceModel =
    new BruteForceModel(uid, centroidsIndexOption, indices)

}