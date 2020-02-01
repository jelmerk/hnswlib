package com.github.jelmerk.spark.knn.hnsw

import com.github.jelmerk.knn.scalalike.SparseVector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import com.github.jelmerk.knn.scalalike.hnsw._
import com.github.jelmerk.spark.knn._
import org.apache.spark.rdd.RDD


/**
  * Companion class for SHnswModel.
  */
object SHnswModel extends MLReadable[SHnswModel] {

  private[hnsw] class SHnswModelReader extends KnnModelReader[
    SHnswModel,
    String,
    SparseVector[Array[Float]],
    SparseVectorIndexItem,
    HnswIndex[String, SparseVector[Array[Float]],
    SparseVectorIndexItem, Float]] {

    override protected def createModel(uid: String, indices: RDD[(Int, (HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float], String, SparseVector[Array[Float]]))]): SHnswModel =
      new SHnswModel(uid, indices)
  }

  override def read: MLReader[SHnswModel] = new SHnswModelReader

}

/**
  * Model produced by a `Hnsw`.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  */
class SHnswModel private[hnsw](override val uid: String,
                indices: RDD[(Int, (HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float], String, SparseVector[Array[Float]]))])
  extends SparseVectorKnnModel[SHnswModel, HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]](uid, indices)
    with MLWritable with HnswModelParams {

  override def copy(extra: ParamMap): SHnswModel = {
    val copied = new SHnswModel(uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override private[knn] def transformIndex(index: HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]): Unit =
    index.ef = getEf

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  override def write: MLWriter = new KnnModelWriter[SHnswModel, String, SparseVector[Array[Float]], SparseVectorIndexItem, HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]](this)
}

/**
  * Nearest neighbor search using the approximative hnsw algorithm.
  *
  * @param uid identifier
  */
class SHnsw(override val uid: String)
  extends SparseVectorKnnAlgorithmBase[SHnswModel, HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]](uid)
    with HnswParams {

  def this() = this(Identifiable.randomUID("hnsw"))

  /** @group setParam */
  def setM(value: Int): this.type = set(m, value)

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  /** @group setParam */
  def setEfConstruction(value: Int): this.type = set(efConstruction, value)

  override def createIndex(maxItemCount: Int): HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float] =
    HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float](
      distanceFunction = distanceFunctionByName(getDistanceFunction),
      maxItemCount = maxItemCount,
      m = getM,
      ef = getEf,
      efConstruction = getEfConstruction
    )

  override def createModel(uid: String,
                           indices: RDD[(Int, (HnswIndex[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float], String, SparseVector[Array[Float]]))]): SHnswModel =
    new SHnswModel(uid, indices)

}

