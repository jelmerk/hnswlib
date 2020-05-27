package com.github.jelmerk.spark.linalg.functions

import com.github.jelmerk.knn.scalalike._
import org.apache.spark.ml.linalg.DenseVector

object DenseVectorDistanceFunctions {

  /**
   * Calculates the cosine distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Cosine distance between u and v.
   */
  def cosineDistance(u: DenseVector, v: DenseVector): Double = doubleCosineDistance(u.values, v.values)

  /**
   * Calculates the inner product.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Cosine distance between u and v.
   */
  def innerProduct(u: DenseVector, v: DenseVector): Double = doubleInnerProduct(u.values, v.values)

  /**
   * Calculates the Bray Curtis distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Bray Curtis distance between u and v.
   */
  def brayCurtisDistance(u: DenseVector, v: DenseVector): Double = doubleBrayCurtisDistance(u.values, v.values)

  /**
   * Calculates the canberra distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Canberra distance between u and v.
   */
  def canberraDistance(u: DenseVector, v: DenseVector): Double = doubleCanberraDistance(u.values, v.values)

  /**
   * Calculates the correlation distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Correlation distance between u and v.
   */
  def correlationDistance(u: DenseVector, v: DenseVector): Double = doubleCorrelationDistance(u.values, v.values)

  /**
   * Calculates the euclidean distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Euclidean distance between u and v.
   */
  def euclideanDistance(u: DenseVector, v: DenseVector): Double = doubleEuclideanDistance(u.values, v.values)

  /**
   * Calculates the manhattan distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Manhattan distance between u and v.
   */
  def manhattanDistance(u: DenseVector, v: DenseVector): Double = doubleManhattanDistance(u.values, v.values)

}
