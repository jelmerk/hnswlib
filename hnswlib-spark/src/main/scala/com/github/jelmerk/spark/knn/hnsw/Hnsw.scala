package com.github.jelmerk.spark.knn.hnsw

import com.github.jelmerk.knn.scalalike.{DistanceFunction, Item}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import com.github.jelmerk.knn.scalalike.hnsw._
import com.github.jelmerk.spark.knn._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vector

private[hnsw] trait HnswParams extends KnnAlgorithmParams with HnswModelParams {

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
    * Has the same meaning as ef, but controls the index time / index precision.
    * Default: 200
    *
    * @group param
    */
  val efConstruction = new IntParam(this, "efConstruction",
    "has the same meaning as ef, but controls the index time / index precision", ParamValidators.gt(0))

  /** @group getParam */
  def getEfConstruction: Int = $(efConstruction)

  setDefault(m -> 16, efConstruction -> 200)
}

/**
  * Common params for Hnsw and HnswModel.
  */
private[hnsw] trait HnswModelParams extends KnnModelParams {

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

  setDefault(ef -> 10)
}

/**
  * Defines the index type.
  */
private[hnsw] trait HswIndexSupport extends IndexSupport {

  override protected type TIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] =
    HnswIndex[TId, TVector, TItem, TDistance]

}

/**
  * Companion class for HnswModel.
  */
object HnswModel extends MLReadable[HnswModel]  {

  private[hnsw] class HnswModelReader extends KnnModelReader[HnswModel] with HswIndexSupport {

    override protected def createModel(uid: String, indices: Either[RDD[(Int, (HnswIndex[String, Array[Float], VectorIndexItemDense, Float], String, Array[Float]))],
                                                                    RDD[(Int, (HnswIndex[String, Vector, VectorIndexItemSparse, Float], String, Vector))]]): HnswModel =
      new HnswModel(uid, indices)
  }

  override def read: MLReader[HnswModel] = new HnswModelReader

}

/**
  * Model produced by a `Hnsw`.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  */
class HnswModel private[hnsw](override val uid: String,
                              override val indices: Either[RDD[(Int, (HnswIndex[String, Array[Float], VectorIndexItemDense, Float], String, Array[Float]))],
                                                           RDD[(Int, (HnswIndex[String, Vector, VectorIndexItemSparse, Float], String, Vector))]])
  extends KnnModel[HnswModel](uid) with MLWritable with HnswModelParams with HswIndexSupport {

  override def copy(extra: ParamMap): HnswModel = {
    val copied = new HnswModel(uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override private[knn] def transformIndex[TVector, TItem <: Item[String, TVector] with Product](index: HnswIndex[String, TVector, TItem, Float]): Unit =
    index.ef = getEf

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  override def write: MLWriter = new KnnModelWriter[HnswModel](this)
}

/**
  * Nearest neighbor search using the approximative hnsw algorithm.
  *
  * @param uid identifier
  */
class Hnsw(override val uid: String) extends KnnAlgorithm[HnswModel](uid) with HnswParams with HswIndexSupport {

  def this() = this(Identifiable.randomUID("hnsw"))

  /** @group setParam */
  def setM(value: Int): this.type = set(m, value)

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  /** @group setParam */
  def setEfConstruction(value: Int): this.type = set(efConstruction, value)

  override protected def createIndex[TVector, TItem <: Item[String, TVector] with Product]
    (dimensions: Int, maxItemCount: Int, distanceFunction: DistanceFunction[TVector, Float])
      : HnswIndex[String, TVector, TItem, Float] = HnswIndex[String, TVector, TItem, Float](
        dimensions,
        distanceFunction = distanceFunction,
        maxItemCount = maxItemCount,
        m = getM,
        ef = getEf,
        efConstruction = getEfConstruction
      )

  override protected def createModel(uid: String, indices: Either[RDD[(Int, (HnswIndex[String, Array[Float], VectorIndexItemDense, Float], String, Array[Float]))],
                                                                  RDD[(Int, (HnswIndex[String, Vector, VectorIndexItemSparse, Float], String, Vector))]]): HnswModel =
    new HnswModel(uid, indices)
}

