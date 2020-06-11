package com.github.jelmerk.spark.linalg.functions

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.FunSuite

import scala.util.Random

class SparseVectorDistanceFunctionsSpec extends FunSuite {

  private implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.001)
  private val random = new Random(1000L)


  test("produce the same result as dense vector functions") {

    // the dense functions should be well tested so lets just compare the results of the spare functions with the dense counterparts

    for (_ <- 1 to 100) {

      val a = createRandomVector()
      val b = createRandomVector()

      assert(DenseVectorDistanceFunctions.innerProduct(a.toDense, b.toDense) === SparseVectorDistanceFunctions.innerProductDistance(a.toSparse, b.toSparse))
      assert(DenseVectorDistanceFunctions.cosineDistance(a.toDense, b.toDense) === SparseVectorDistanceFunctions.cosineDistance(a.toSparse, b.toSparse))
      assert(DenseVectorDistanceFunctions.euclideanDistance(a.toDense, b.toDense) === SparseVectorDistanceFunctions.euclideanDistance(a.toSparse, b.toSparse))
      assert(DenseVectorDistanceFunctions.brayCurtisDistance(a.toDense, b.toDense) === SparseVectorDistanceFunctions.brayCurtisDistance(a.toSparse, b.toSparse))
      assert(DenseVectorDistanceFunctions.canberraDistance(a.toDense, b.toDense) === SparseVectorDistanceFunctions.canberraDistance(a.toSparse, b.toSparse))
      assert(DenseVectorDistanceFunctions.manhattanDistance(a.toDense, b.toDense) === SparseVectorDistanceFunctions.manhattanDistance(a.toSparse, b.toSparse))
      assert(DenseVectorDistanceFunctions.correlationDistance(a.toDense, b.toDense) === SparseVectorDistanceFunctions.correlationDistance(a.toSparse, b.toSparse))

    }

  }

  def createRandomVector(): Vector = Vectors.dense(
    Iterator.continually {
      val roll = random.nextInt(10)

      if (roll < 2) 0.0
      else if (roll > 6) random.nextDouble()
      else -random.nextDouble()
    }
    .take(1000)
    .toArray
  )


}

