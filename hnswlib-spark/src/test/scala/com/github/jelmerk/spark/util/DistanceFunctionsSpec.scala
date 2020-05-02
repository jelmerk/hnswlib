package com.github.jelmerk.spark.util

import org.apache.spark.ml.linalg.Vectors
import org.scalatest.FunSuite
import org.scalatest.Matchers._

class DistanceFunctionsSpec extends FunSuite {

  test("calculate inner product") {
    val a = Vectors.sparse(2, Array(0, 1), Array(0.047, 1)).toSparse
    val b = Vectors.sparse(2, Array(0, 1), Array(0.51, 0.86)).toSparse

    DistanceFunctions.innerProductDistance(a, b) should be(0.11602999999999997)
  }

  test("calculate inner product with empty positions") {
    val a = Vectors.sparse(3, Array(1, 2), Array(1, 0.5)).toSparse
    val b = Vectors.sparse(3, Array(0, 1), Array(0.5, 0.9)).toSparse

    DistanceFunctions.innerProductDistance(a, b) should be(0.09999999999999998)
  }

  test("calculate cosine distance") {
    val a = Vectors.sparse(3, Array(0, 1, 2), Array(0.01, 0.02, 0.03)).toSparse
    val b = Vectors.sparse(3, Array(0, 1, 2), Array(0.03, 0.02, 0.01)).toSparse

    DistanceFunctions.cosineDistance(a, b) should be(0.2857142857142858)
  }

  test("calculate cosine distance with empty positions") {

    val a = Vectors.sparse(3, Array(1, 2), Array(1, 0.5)).toSparse
    val b = Vectors.sparse(3, Array(0, 1), Array(0.5, 0.9)).toSparse

    DistanceFunctions.cosineDistance(a, b) should be(0.21812996302647503)
  }

}

