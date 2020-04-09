package com.github.jelmerk.spark.knn

import org.apache.spark.ml.linalg.SparseVector

object DistanceFunctions {

  def cosineDistance(u: SparseVector, v: SparseVector): Double = {
    val denom = norm(u) * norm(v)
    val dot = innerProduct(u, v)

    if (denom == 0d) 1d
    else 1 - dot / denom
  }

  def innerProductDistance(u: SparseVector, v: SparseVector): Double = 1 - innerProduct(u, v)

  private def norm(u: SparseVector): Double = math.sqrt(u.values.map(v => v * v).sum)

  private def innerProduct(u: SparseVector, v: SparseVector): Double = {
    val uIndices = u.indices
    val vIndices = v.indices

    val uValues = u.values
    val vValues = v.values

    var dot = 0d

    var i = 0
    var j = 0

    while(i < uIndices.length && j < vIndices.length) {
      if (uIndices(i) < vIndices(j)) {
        i += 1
      } else if (uIndices(i) > vIndices(j)) {
        j += 1
      } else {
        dot += uValues(i) * vValues(j)
        i += 1
        j += 1
      }
    }
    dot
  }

}
