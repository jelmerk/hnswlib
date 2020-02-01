package com.github.jelmerk.spark.knn

import scala.reflect.ClassTag
import scala.util.Try

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import com.github.jelmerk.knn.scalalike.{Index, Item}
import com.github.jelmerk.spark.util.Udfs.{doubleArrayToFloatArray, vectorToFloatArray}
import org.apache.spark.ml.Model
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, StringType}
import com.github.jelmerk.knn.scalalike._

abstract class DenseVectorKnnAlgorithmBase[
  TModel <: Model[TModel],
  TId : TypeTag,
  TVector : TypeTag,
  TItem <: Item[TId, TVector] with Product : TypeTag,
  TIndex <: Index[TId, TVector, TItem, Float]
]
(uid: String)(implicit ev: ClassTag[TItem]) extends KnnAlgorithm[TModel, TId, TVector, TItem, TIndex](uid) {

  private[knn] override def readItems(dataset: Dataset[_]): Dataset[TItem] = {
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
      ).as[TItem]
  }

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
