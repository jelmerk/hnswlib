package com.github.jelmerk.spark

import com.github.jelmerk.knn.{SparseVector, DistanceFunctions => JDistanceFunctions}
import com.github.jelmerk.spark.knn.DistanceFunctions
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.FunSuite
import org.scalatest.Matchers._

class DistanceFunctionsSpec extends FunSuite {

  test("calculate inner product") {



    println(JDistanceFunctions.FLOAT_INNER_PRODUCT.distance(
      Array(0.047f, 1f), Array(0.51f, 0.86f)
    ))


    println(JDistanceFunctions.FLOAT_SPARSE_VECTOR_INNER_PRODUCT.distance(
      new SparseVector(Array(0, 1), Array(0.047f, 1f)),
      new SparseVector(Array(0, 1), Array(0.51f, 0.86f))
    ))


    val result = DistanceFunctions.innerProduct(
      Vectors.sparse(2, Array(0, 1), Array(0.047f, 1f)),
      Vectors.sparse(2, Array(0, 1), Array(0.51f, 0.86f))
    )

    println(result)

    result should be(0.116029985f)
  }

}
