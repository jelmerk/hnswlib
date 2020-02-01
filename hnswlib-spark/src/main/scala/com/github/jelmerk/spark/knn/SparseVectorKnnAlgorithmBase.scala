package com.github.jelmerk.spark.knn

import scala.util.Try

import com.github.jelmerk.knn.scalalike.Index
import com.github.jelmerk.spark.util.Udfs.sparseVectorToHnswLibSparseVector
import org.apache.spark.ml.Model
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType
import com.github.jelmerk.knn.scalalike._
import org.apache.spark.rdd.RDD

/**
 * Item in an nearest neighbor search index
 *
 * @param id item identifier
 * @param vector item vector
 */
private[knn] case class SparseVectorIndexItem(id: String, vector: SparseVector[Array[Float]]) extends Item[String, SparseVector[Array[Float]]]

private[knn] trait SparseVectorIndexItemsReader extends KnnModelParams {

  /**
   * Read items from dataset.
   *
   * @param dataset the dataset
   * @return dataset of items
   */
  def readItems(dataset: Dataset[_]): Dataset[SparseVectorIndexItem] = {
    import dataset.sparkSession.implicits._

    dataset
      .select(
        col(getIdentifierCol).cast(StringType).alias("id"),
        sparseVectorToHnswLibSparseVector(col(getVectorCol)).alias("vector")
      ).as[SparseVectorIndexItem]
  }
}

private[knn] abstract class SparseVectorKnnModel[TModel <: Model[TModel],
                                                 TIndex <: Index[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]]
      (uid: String, indices: RDD[(Int, (TIndex, String, SparseVector[Array[Float]]))])
    extends KnnModel[TModel, String, SparseVector[Array[Float]], SparseVectorIndexItem, TIndex](uid, indices)
      with SparseVectorIndexItemsReader

private[knn] abstract class SparseVectorKnnAlgorithmBase[TModel <: Model[TModel],
                                                         TIndex <: Index[String, SparseVector[Array[Float]], SparseVectorIndexItem, Float]]
    (uid: String)
  extends KnnAlgorithm[TModel, String, SparseVector[Array[Float]], SparseVectorIndexItem, TIndex](uid)
    with SparseVectorIndexItemsReader {

  private[knn] def distanceFunctionByName(name: String): DistanceFunction[SparseVector[Array[Float]], Float] = name match {
    case "inner-product" => floatSparseVectorInnerProduct
    case value =>
      Try(Class.forName(value).getDeclaredConstructor().newInstance())
        .toOption
        .collect { case f: DistanceFunction[SparseVector[Array[Float]] @unchecked, Float @unchecked] => f }
        .getOrElse(throw new IllegalArgumentException(s"$value is not a valid distance function."))
  }
}
