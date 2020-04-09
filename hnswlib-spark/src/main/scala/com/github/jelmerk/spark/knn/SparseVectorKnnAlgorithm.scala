package com.github.jelmerk.spark.knn

import com.github.jelmerk.knn.scalalike._
import org.apache.spark.ml.Model
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.SparseVector

import scala.util.Try

// TODO com.github.jelmerk.knn.SparseVector is not a case class so spark does not know how to serialize it but
//  the standard vector uses up a lot of space

/**
  * Item in an nearest neighbor search index
  *
  * @param id item identifier
  * @param vector item vector
  */
private[knn] case class SparseVectorIndexItem(id: String, vector: Vector) extends Item[String, Vector] {
  override def dimensions: Int = vector.size
}

private[knn] abstract class SparseVectorKnnModel[TModel <: Model[TModel], TIndex <: Index[String, Vector, SparseVectorIndexItem, Float]](uid: String, indices: RDD[(Int, (TIndex, String, Vector))])
  extends KnnModel[TModel, String, Vector, SparseVectorIndexItem, TIndex](uid, indices) {
  /**
    * Read queries from the passed in dataset.
    *
    * @param dataset the dataset to read the items from
    * @return dataset of item
    */
  override private[knn] def readQueries(dataset: Dataset[_]): Dataset[Query[Vector]] = {
    import dataset.sparkSession.implicits._

    dataset
      .select(
        col(getIdentifierCol).cast(StringType).alias("id"),
        col(getVectorCol).alias("vector")
      ).as[Query[Vector]]
  }
}

private[knn] abstract class SparseVectorKnnAlgorithm[TModel <: Model[TModel], TIndex <: Index[String, Vector, SparseVectorIndexItem, Float]](override val uid: String)
  extends KnnAlgorithm[TModel, String, Vector, SparseVectorIndexItem, TIndex](uid) {

  override def readIndexItems(dataset: Dataset[_]): Dataset[SparseVectorIndexItem] = {

    import dataset.sparkSession.implicits._

    dataset
      .select(
        col(getIdentifierCol).cast(StringType).alias("id"),
        col(getVectorCol).alias("vector")
      ).as[SparseVectorIndexItem]
  }

  protected def distanceFunctionByName(name: String): DistanceFunction[Vector, Float] = name match {
    case "cosine" => (u: Vector, v: Vector) => DistanceFunctions.cosineDistance(u.asInstanceOf[SparseVector], v.asInstanceOf[SparseVector]).toFloat
    case "inner-product" => (u: Vector, v: Vector) => DistanceFunctions.innerProductDistance(u.asInstanceOf[SparseVector], v.asInstanceOf[SparseVector]).toFloat
    case value =>
      Try(Class.forName(value).getDeclaredConstructor().newInstance())
        .toOption
        .collect { case f: DistanceFunction[Vector @unchecked, Float @unchecked] => f }
        .getOrElse(throw new IllegalArgumentException(s"$value is not a valid distance function."))
  }

}
