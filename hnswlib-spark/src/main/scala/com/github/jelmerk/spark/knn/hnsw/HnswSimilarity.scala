package com.github.jelmerk.spark.knn.hnsw

import java.io.InputStream

import scala.reflect.runtime.universe._
import scala.reflect.ClassTag
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import com.github.jelmerk.knn.scalalike.{DistanceFunction, Item}
import com.github.jelmerk.knn.scalalike.hnsw._
import com.github.jelmerk.spark.knn._

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
  * Companion class for HnswSimilarityModel.
  */
object HnswSimilarityModel extends MLReadable[HnswSimilarityModel]  {

  private[hnsw] class HnswModelReader extends KnnModelReader[HnswSimilarityModel] {

    override protected def createModel[
      TId: TypeTag,
      TVector: TypeTag,
      TItem <: Item[TId, TVector] with Product: TypeTag,
      TDistance : TypeTag
    ](uid: String, indices: RDD[(Int, String)])
      (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance]) : HnswSimilarityModel =
        new HnswSimilarityModelImpl[TId, TVector, TItem, TDistance](uid, indices)
  }

  override def read: MLReader[HnswSimilarityModel] = new HnswModelReader

}

/**
  * Model produced by `HnswSimilarity`.
  */
abstract class HnswSimilarityModel extends KnnModelBase[HnswSimilarityModel] with HnswModelParams with MLWritable {

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

}

private[knn] class HnswSimilarityModelImpl[
  TId : TypeTag,
  TVector : TypeTag,
  TItem <: Item[TId, TVector] with Product : TypeTag,
  TDistance : TypeTag
](override val uid: String, private[knn] val indices: RDD[(Int, String)])
  (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance])
    extends HnswSimilarityModel with KnnModelOps[HnswSimilarityModel, TId, TVector, TItem, TDistance, HnswIndex[TId, TVector, TItem, TDistance]] {

  override def transform(dataset: Dataset[_]): DataFrame = typedTransform(dataset)

  override def copy(extra: ParamMap): HnswSimilarityModel = {
    val copied = new HnswSimilarityModelImpl[TId, TVector, TItem, TDistance](uid, indices)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new KnnModelWriter[HnswSimilarityModel, TId, TVector, TItem, TDistance, HnswIndex[TId, TVector, TItem, TDistance]](this)

  override protected def loadIndex(in: InputStream): HnswIndex[TId, TVector, TItem, TDistance] = {
    val index = HnswIndex.load[TId, TVector, TItem, TDistance](in)
    index.ef = getEf
    index
  }
}


/**
  * Nearest neighbor search using the approximative hnsw algorithm.
  *
  * @param uid identifier
  */
class HnswSimilarity(override val uid: String) extends KnnAlgorithm[HnswSimilarityModel](uid) with HnswParams {

  override protected type TIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] = HnswIndex[TId, TVector, TItem, TDistance]

  def this() = this(Identifiable.randomUID("hnsw"))

  /** @group setParam */
  def setM(value: Int): this.type = set(m, value)

  /** @group setParam */
  def setEf(value: Int): this.type = set(ef, value)

  /** @group setParam */
  def setEfConstruction(value: Int): this.type = set(efConstruction, value)

  override protected def createIndex[TId, TVector, TItem <: Item[TId, TVector] with Product, TDistance]
    (dimensions: Int, maxItemCount: Int, distanceFunction: DistanceFunction[TVector, TDistance])(implicit distanceOrdering: Ordering[TDistance])
      : HnswIndex[TId, TVector, TItem, TDistance] =
           HnswIndex[TId, TVector, TItem, TDistance](
            dimensions,
            distanceFunction,
            maxItemCount,
            getM,
            getEf,
            getEfConstruction
          )

  override protected def createModel[
    TId: TypeTag,
    TVector: TypeTag,
    TItem <: Item[TId, TVector] with Product: TypeTag,
    TDistance : TypeTag
  ](uid: String, indices: RDD[(Int, String)])
    (implicit evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance]) : HnswSimilarityModel =
      new HnswSimilarityModelImpl[TId, TVector, TItem, TDistance](uid, indices)
}

