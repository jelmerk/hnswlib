package com.github.jelmerk.spark.linalg.functions

import com.github.jelmerk.knn.scalalike._
import org.apache.spark.ml.linalg.DenseVector

object DenseVectorDistanceFunctions {

  def cosineDistance(u: DenseVector, v: DenseVector): Double = doubleCosineDistance(u.values, v.values)

  def innerProduct(u: DenseVector, v: DenseVector): Double = doubleInnerProduct(u.values, v.values)

  def brayCurtisDistance(u: DenseVector, v: DenseVector): Double = doubleBrayCurtisDistance(u.values, v.values)

  def canberraDistance(u: DenseVector, v: DenseVector): Double = doubleCanberraDistance(u.values, v.values)

  def correlationDistance(u: DenseVector, v: DenseVector): Double = doubleCorrelationDistance(u.values, v.values)

  def euclideanDistance(u: DenseVector, v: DenseVector): Double = doubleEuclideanDistance(u.values, v.values)

  def manhattanDistance(u: DenseVector, v: DenseVector): Double = doubleManhattanDistance(u.values, v.values)

}
