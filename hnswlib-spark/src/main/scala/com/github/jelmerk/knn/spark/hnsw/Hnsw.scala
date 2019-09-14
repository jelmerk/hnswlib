package com.github.jelmerk.knn.spark.hnsw

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.hnsw._
import com.github.jelmerk.knn.spark._
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD


trait HnswParams extends KnnAlgorithmParams with KnnModelParams {

  /**
    * The number of bi-directional links created for every new element during construction.
    *
    * Default: 16
    *
    * @group param
    */
  val m = new IntParam(this, "m",
    "number of bi-directional links created for every new element during construction", ParamValidators.gt(0))

  /** @group getParam */
  def getM: Int = $(m)

  /**
    * Size of the dynamic list for the nearest neighbors (used during the search).
    * Default: 10
    *
    * @group param
    */
  val ef = new IntParam(this, "ef",
    "size of the dynamic list for the nearest neighbors (used during the search)", ParamValidators.gt(0))

  /** @group getParam */
  def getEf: Int = $(ef)

  /**
    * Has the same meaning as ef, but controls the index time / index precision.
    * Default: 200
    *
    * @group param
    */
  val efConstruction = new IntParam(this, "efConstruction",
    "has the same meaning as ef, but controls the index time / index precision", ParamValidators.gt(0))

  /** @group getParam */
  def getEfConstruction: Int = $(efConstruction)


  setDefault(m -> 16, ef -> 10, efConstruction -> 200, numPartitions -> 1, k -> 5,
    neighborsCol -> "neighbors", identifierCol -> "id", vectorCol -> "vector", distanceFunction -> "cosine")
}


class HnswModel(override val uid: String,
                numPartitions: Int,
                partitioner: Partitioner,
                indices: RDD[(Int, Index[String, Array[Float], IndexItem, Float])])
  extends KnnModel[HnswModel](uid, numPartitions, partitioner, indices) {


  override def copy(extra: ParamMap): HnswModel = {
    val copied = new HnswModel(uid, numPartitions, partitioner, indices)
    copyValues(copied, extra).setParent(parent)
  }

}

class Hnsw(override val uid: String) extends KnnAlgorithm[HnswModel](uid) with HnswParams {

  def this() = this(Identifiable.randomUID("hnsw"))

  /** @group setParam */
  def setM(value: Int): this.type = set(m, value)

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  /** @group setParam */
  def setEfConstruction(value: Int): this.type = set(efConstruction, value)

  override def createIndex(maxItemCount: Int): Index[String, Array[Float], IndexItem, Float] =
    HnswIndex[String, Array[Float], IndexItem, Float](
      distanceFunction = distanceFunctionByName(getDistanceFunction),
      maxItemCount = maxItemCount,
      m = getM,
      ef = getEf,
      efConstruction = getEfConstruction
    )

  override def createModel(uid: String,
                           numPartitions: Int,
                           partitioner: Partitioner,
                           indices: RDD[(Int, Index[String, Array[Float], IndexItem, Float])]): HnswModel =
    new HnswModel(uid, numPartitions, partitioner, indices)

}

