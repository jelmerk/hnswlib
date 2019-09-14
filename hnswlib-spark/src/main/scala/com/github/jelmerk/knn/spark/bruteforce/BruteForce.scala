package com.github.jelmerk.knn.spark.bruteforce

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.bruteforce.BruteForceIndex
import com.github.jelmerk.knn.spark._
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD

class BruteForceModel(override val uid: String,
                      numPartitions: Int,
                      partitioner: Partitioner,
                      indices: RDD[(Int, Index[String, Array[Float], IndexItem, Float])])
  extends KnnModel[BruteForceModel](uid, numPartitions, partitioner, indices) {


  override def copy(extra: ParamMap): BruteForceModel = {
    val copied = new BruteForceModel(uid, numPartitions, partitioner, indices)
    copyValues(copied, extra).setParent(parent)
  }

}

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