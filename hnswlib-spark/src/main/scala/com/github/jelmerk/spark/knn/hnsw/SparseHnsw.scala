package com.github.jelmerk.spark.knn.hnsw

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import com.github.jelmerk.knn.scalalike.hnsw._
import com.github.jelmerk.spark.knn._
import org.apache.spark.rdd.RDD

/**
  * Companion class for SparseHnswModel.
  */
object SparseHnswModel extends MLReadable[SparseHnswModel] {

  private[hnsw] class SparseHnswModelReader extends KnnModelReader[SparseHnswModel, String, Vector, SparseVectorIndexItem, HnswIndex[String, Vector, SparseVectorIndexItem, Float]] {
    override protected def createModel(uid: String, indices: RDD[(Int, (HnswIndex[String, Vector, SparseVectorIndexItem, Float], String, Vector))]): SparseHnswModel =
      new SparseHnswModel(uid, indices)
  }

  override def read: MLReader[SparseHnswModel] = new SparseHnswModelReader

}

/**
  * Model produced by a `SparseHnsw`.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  */
class SparseHnswModel private[hnsw](override val uid: String,
                                    indices: RDD[(Int, (HnswIndex[String, Vector, SparseVectorIndexItem, Float], String, Vector))])
  extends SparseVectorKnnModel[SparseHnswModel, HnswIndex[String, Vector, SparseVectorIndexItem, Float]](uid, indices) with MLWritable with HnswModelParams {

  override def copy(extra: ParamMap): SparseHnswModel = {
    val copied = new SparseHnswModel(uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override private[knn] def transformIndex(index: HnswIndex[String, Vector, SparseVectorIndexItem, Float]): Unit =
    index.ef = getEf

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  override def write: MLWriter = new KnnModelWriter[SparseHnswModel, String, Vector, SparseVectorIndexItem, HnswIndex[String, Vector, SparseVectorIndexItem, Float]](this)
}

/**
  * Nearest neighbor search using the approximative hnsw algorithm.
  *
  * @param uid identifier
  */
class SparseHnsw(override val uid: String) extends SparseVectorKnnAlgorithm[SparseHnswModel, HnswIndex[String, Vector, SparseVectorIndexItem, Float]](uid)
  with HnswParams {

  def this() = this(Identifiable.randomUID("hnsw"))

  /** @group setParam */
  def setM(value: Int): this.type = set(m, value)

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  /** @group setParam */
  def setEfConstruction(value: Int): this.type = set(efConstruction, value)

  override def createIndex(dimensions: Int, maxItemCount: Int): HnswIndex[String, Vector, SparseVectorIndexItem, Float] =
    HnswIndex[String, Vector, SparseVectorIndexItem, Float](
      dimensions,
      distanceFunction = distanceFunctionByName(getDistanceFunction),
      maxItemCount = maxItemCount,
      m = getM,
      ef = getEf,
      efConstruction = getEfConstruction
    )

  override def createModel(uid: String,
                           indices: RDD[(Int, (HnswIndex[String, Vector, SparseVectorIndexItem, Float], String, Vector))]): SparseHnswModel =
    new SparseHnswModel(uid, indices)

}

