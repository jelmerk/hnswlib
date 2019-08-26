package com.github.jelmerk.knn

import com.github.jelmerk.knn.{ DistanceFunctions => JDistanceFunctions }
//import com.github.jelmerk.knn.spark.{ DistanceFunctions => SparkDistanceFunctions }
//import org.apache.spark.ml.linalg.Vector

package object scalalike {

  /**
    * Item that can be indexed
    *
    * @tparam TId The type of the vector to perform distance calculation on
    * @tparam TVector Type of the vector to perform distance calculation on
    */
  type Item[TId, TVector] = com.github.jelmerk.knn.Item[TId, TVector]

  /**
    * Result of a nearest neighbour search
    * @tparam TItem type of the item returned
    * @tparam TDistance type of the distance returned by the configured distance function
    */
  type SearchResult[TItem, TDistance] = com.github.jelmerk.knn.SearchResult[TItem, TDistance]

  /**
    * Calculates distance between 2 vectors
    */
  type DistanceFunction[TVector, TDistance] = (TVector, TVector) => TDistance

  /**
    * Called periodically to report on progress during indexing. The first argument is the number of records processed.
    * The second argument is the total number of record to index as part of this operation.
    */
  type ProgressListener = (Int, Int) => Unit

  /**
    * Calculates cosine distance.
    */
  val floatCosineDistance: DistanceFunction[Array[Float], Float] = JDistanceFunctions.FLOAT_COSINE_DISTANCE.distance

  /**
    * Calculates inner product.
    */
  val floatInnerProduct: DistanceFunction[Array[Float], Float] = JDistanceFunctions.FLOAT_INNER_PRODUCT.distance

  /**
    * Calculates cosine distance.
    */
  val doubleCosineDistance: DistanceFunction[Array[Double], Double] = JDistanceFunctions.DOUBLE_COSINE_DISTANCE.distance

  /**
    * Calculates inner product.
    */
  val doubleInnerProduct: DistanceFunction[Array[Double], Double] = JDistanceFunctions.DOUBLE_INNER_PRODUCT.distance
//
//  /**
//    * Calculates cosine distance.
//    */
//  val denseVectorCosineDistance: DistanceFunction[Vector, Double] = SparkDistanceFunctions.DENSE_VECTOR_COSINE_DISTANCE.distance
//
//  /**
//    * Calculates inner product.
//    */
//  val denseVectorInnerProduct: DistanceFunction[Vector, Double] = SparkDistanceFunctions.DENSE_VECTOR_INNER_PRODUCT.distance
//
//  /**
//    * Calculates cosine distance.
//    */
//  val sparseVectorCosineDistance: DistanceFunction[Vector, Double] = SparkDistanceFunctions.SPARSE_VECTOR_COSINE_DISTANCE.distance
//
//  /**
//    * Calculates inner product.
//    */
//  val sparseVectorInnerProduct: DistanceFunction[Vector, Double] = SparkDistanceFunctions.SPARSE_VECTOR_INNER_PRODUCT.distance

}
