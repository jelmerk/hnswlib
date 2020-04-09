package com.github.jelmerk.spark.knn

import org.apache.spark.ml.linalg.{SparseVector, Vector}

object DistanceFunctions {

  def innerProduct(u: Vector, v: Vector): Float = {

    val uIndices = u.asInstanceOf[SparseVector].indices
    val vIndices = u.asInstanceOf[SparseVector].indices

    val uValues = u.asInstanceOf[SparseVector].values
    val vValues = v.asInstanceOf[SparseVector].values

    var dot = 0.0

    var i = 0
    var j = 0

    while(i < uIndices.length && j < uIndices.length) {
      if (uIndices(i) < vIndices(j)) {
        i += 1
      } else if (uIndices(i) < vIndices(j)) {
        j += 1
      } else  {
        dot += uValues(i) * vValues(j)
        i += 1
        j += 1
      }
    }

    (1 - dot).toFloat

  }

}
