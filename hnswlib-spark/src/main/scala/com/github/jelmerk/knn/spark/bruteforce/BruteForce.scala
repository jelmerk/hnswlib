package com.github.jelmerk.knn.spark.bruteforce

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.knn.spark._
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD

/**
  * Model produced by a `BruteForce`.
  *
  * @param uid identifier
  * @param numPartitions how many partitions
  * @param partitioner the partitioner used to parition the data
  * @param indices rdd that holds the indices that are used to do the search
  */
class BruteForceModel(override val uid: String,
                      numPartitions: Int,
                      partitioner: Partitioner,
                      indices: RDD[(Int, Index[String, Array[Float], IndexItem, Float])]
                     )
  extends KnnModel[BruteForceModel](uid, numPartitions, partitioner, indices) {


  override def copy(extra: ParamMap): BruteForceModel = {
    val copied = new BruteForceModel(uid, numPartitions, partitioner, indices)
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
                           numPartitions: Int,
                           partitioner: Partitioner,
                           indices: RDD[(Int, Index[String, Array[Float], IndexItem, Float])]): BruteForceModel =
    new BruteForceModel(uid, numPartitions, partitioner, indices)

}