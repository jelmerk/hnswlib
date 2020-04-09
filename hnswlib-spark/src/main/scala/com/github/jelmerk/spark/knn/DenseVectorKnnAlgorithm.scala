package com.github.jelmerk.spark.knn

import com.github.jelmerk.knn.scalalike._
import org.apache.spark.ml.Model
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, StringType}
import com.github.jelmerk.spark.functions._
import org.apache.spark.rdd.RDD

import scala.util.Try

/**
  * Item in an nearest neighbor search index
  *
  * @param id item identifier
  * @param vector item vector
  */
private[knn] case class DenseVectorIndexItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]] {
  override def dimensions: Int = vector.length
}

private[knn] abstract class DenseVectorKnnModel[TModel <: Model[TModel], TIndex <: Index[String, Array[Float], DenseVectorIndexItem, Float]]
  (uid: String, indices: RDD[(Int, (TIndex, String, Array[Float]))])
    extends KnnModel[TModel, String, Array[Float], DenseVectorIndexItem, TIndex](uid, indices) {
  /**
    * Read queries from the passed in dataset.
    *
    * @param dataset the dataset to read the items from
    * @return dataset of item
    */
  override private[knn] def readQueries(dataset: Dataset[_]): Dataset[Query[Array[Float]]] = {
    import dataset.sparkSession.implicits._

    val vectorCol = dataset.schema(getVectorCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getVectorCol))
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getVectorCol))
      case _ => col(getVectorCol)
    }

    dataset
      .select(
        col(getIdentifierCol).cast(StringType).as("id"),
        vectorCol.as("vector")
      ).as[Query[Array[Float]]]
  }
}


private[knn] abstract class DenseVectorKnnAlgorithm[TModel <: Model[TModel], TIndex <: Index[String, Array[Float], DenseVectorIndexItem, Float]](override val uid: String)
  extends KnnAlgorithm[TModel, String, Array[Float], DenseVectorIndexItem, TIndex](uid) {

  override def readIndexItems(dataset: Dataset[_]): Dataset[DenseVectorIndexItem] = {

    import dataset.sparkSession.implicits._

    val vectorCol = dataset.schema(getVectorCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getVectorCol))
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getVectorCol))
      case _ => col(getVectorCol)
    }

    dataset
      .select(
        col(getIdentifierCol).cast(StringType).as("id"),
        vectorCol.as("vector")
      ).as[DenseVectorIndexItem]
  }

  protected def distanceFunctionByName(name: String): DistanceFunction[Array[Float], Float] = name match {
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
