package com.github.jelmerk.spark.knn

import scala.util.Try

import com.github.jelmerk.knn.scalalike.Index
import com.github.jelmerk.spark.util.Udfs.{doubleArrayToFloatArray, vectorToFloatArray}
import org.apache.spark.ml.Model
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, StringType}
import com.github.jelmerk.knn.scalalike._
import org.apache.spark.rdd.RDD

/**
 * Item in an nearest neighbor search index
 *
 * @param id item identifier
 * @param vector item vector
 */
private[knn] case class DenseVectorIndexItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]]

private[knn] trait DenseVectorIndexItemsReader extends KnnModelParams {

  /**
   * Read items from dataset.
   *
   * @param dataset the dataset
   * @return dataset of items
   */
  def readItems(dataset: Dataset[_]): Dataset[DenseVectorIndexItem] = {
    import dataset.sparkSession.implicits._

    val vectorCol = dataset.schema(getVectorCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getVectorCol))
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getVectorCol))
      case _ => col(getVectorCol)
    }

    dataset
      .select(
        col(getIdentifierCol).cast(StringType).alias("id"),
        vectorCol.alias("vector")
      ).as[DenseVectorIndexItem]
  }
}

private[knn] abstract class DenseVectorKnnModel[TModel <: Model[TModel],
                                                TIndex <: Index[String, Array[Float], DenseVectorIndexItem, Float]]
      (uid: String, indices: RDD[(Int, (TIndex, String, Array[Float]))])
    extends KnnModel[TModel, String, Array[Float], DenseVectorIndexItem, TIndex](uid, indices)
      with DenseVectorIndexItemsReader

private[knn] abstract class DenseVectorKnnAlgorithmBase[TModel <: Model[TModel],
                                                        TIndex <: Index[String, Array[Float], DenseVectorIndexItem, Float]]
    (uid: String)
  extends KnnAlgorithm[TModel, String, Array[Float], DenseVectorIndexItem, TIndex](uid)
    with DenseVectorIndexItemsReader {

  private[knn] def distanceFunctionByName(name: String): DistanceFunction[Array[Float], Float] = name match {
    case "bray-curtis" => floatBrayCurtisDistance
    case "canberra" => floatCanberraDistance
    case "correlation" => floatCorrelationDistance
    case "cosine" => floatCosineDistance
    case "euclidean" => floatEuclideanDistance
    case "inner-product" => floatInnerProduct
    case "manhattan" => floatManhattanDistance
    case value =>
      Try(Class.forName(value).getDeclaredConstructor().newInstance())
        .toOption
        .collect { case f: DistanceFunction[Array[Float] @unchecked, Float @unchecked] => f }
        .getOrElse(throw new IllegalArgumentException(s"$value is not a valid distance function."))
  }
}
