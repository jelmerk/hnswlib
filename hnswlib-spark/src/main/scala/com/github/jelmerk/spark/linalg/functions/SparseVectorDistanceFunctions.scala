package com.github.jelmerk.spark.linalg.functions

import org.apache.spark.ml.linalg.SparseVector
import math.{abs, sqrt, pow}

object SparseVectorDistanceFunctions {

  /**
   * Calculates the cosine distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Cosine distance between u and v.
   */
  def cosineDistance(u: SparseVector, v: SparseVector): Double = {
    val denom = norm(u) * norm(v)
    val dot = innerProduct(u, v)

    if (denom == 0.0) 1d
    else 1 - dot / denom
  }

  /**
   * Calculates the inner product.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Cosine distance between u and v.
   */
  def innerProductDistance(u: SparseVector, v: SparseVector): Double = 1 - innerProduct(u, v)

  /**
   * Calculates the Bray Curtis distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Bray Curtis distance between u and v.
   */
  def brayCurtisDistance(u: SparseVector, v: SparseVector): Double = {
    val uIndices = u.indices
    val vIndices = v.indices

    val uValues = u.values
    val vValues = v.values

    var sump = 0.0
    var sumn = 0.0

    var i = 0
    var j = 0

    while(i < uIndices.length && j < vIndices.length) {
      if (uIndices(i) < vIndices(j)) {
        val incr = abs(uValues(i))

        sumn += incr
        sump += incr
        i += 1
      } else if (uIndices(i) > vIndices(j)) {
        val incr = abs(vValues(j))

        sumn += incr
        sump += incr
        j += 1
      } else {
        sumn += abs(uValues(i) - vValues(j))
        sump += abs(uValues(i) + vValues(j))
        i += 1
        j += 1
      }
    }

    while(i < uIndices.length) {
      val incr = abs(uValues(i))

      sumn += incr
      sump += incr
      i += 1
    }

    while(j < vIndices.length) {
      val incr = abs(vValues(j))

      sumn += incr
      sump += incr
      j += 1
    }

    sumn / sump
  }

  /**
   * Calculates the canberra distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Canberra distance between u and v.
   */
  def canberraDistance(u: SparseVector, v: SparseVector): Double = {
    val uIndices = u.indices
    val vIndices = v.indices

    val uValues = u.values
    val vValues = v.values

    var distance = 0.0

    var i = 0
    var j = 0

    while(i < uIndices.length && j < vIndices.length) {
      if (uIndices(i) < vIndices(j)) {
        distance += 1
        i += 1
      } else if (uIndices(i) > vIndices(j)) {
        distance += 1
        j += 1
      } else {
        distance += abs(uValues(i) - vValues(j)) / (abs(uValues(i)) + abs(vValues(j)))
        i += 1
        j += 1
      }
    }

    distance + (uIndices.length - i) + (vIndices.length - j)
  }

  /**
   * Calculates the correlation distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Correlation distance between u and v.
   */
  def correlationDistance(u: SparseVector, v: SparseVector): Double = {

    val x = -u.values.sum / u.size
    val y = -v.values.sum / v.size

    val uIndices = u.indices
    val vIndices = v.indices

    val uValues = u.values
    val vValues = v.values

    var num = 0.0
    var den1 = 0.0
    var den2 = 0.0

    var left = u.size

    var i = 0
    var j = 0

    val absXSquared = abs(pow(x, 2))

    while(i < uIndices.length && j < vIndices.length) {
      if (uIndices(i) < vIndices(j)) {
        num += (uValues(i) + x) * y

        den1 += abs(pow(uValues(i) + x, 2))
        den2 += absXSquared
        left -= 1

        i += 1
      } else if (uIndices(i) > vIndices(j)) {
        num += x * (vValues(j) + y)

        den1 += absXSquared
        den2 += abs(pow(vValues(j) + x, 2))
        left -= 1
        j += 1
      } else {
        num += (uValues(i) + x) * (vValues(j) + y)

        den1 += abs(pow(uValues(i) + x, 2))
        den2 += abs(pow(vValues(j) + x, 2))
        left -= 1

        i += 1
        j += 1
      }
    }

    while(i < uIndices.length) {
      num += (uValues(i) + x) * y

      den1 += abs(pow(uValues(i) + x, 2))
      den2 += absXSquared
      left -= 1

      i += 1
    }

    while(j < vIndices.length) {
      num += x * (vValues(j) + y)

      den1 += absXSquared
      den2 += abs(pow(vValues(j) + x, 2))

      j += 1
    }

    num += (x * y) * left
    den1 += absXSquared * left
    den2 += absXSquared * left

    1 - (num / (sqrt(den1) * sqrt(den2)))
  }

  /**
   * Calculates the euclidean distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Euclidean distance between u and v.
   */
  def euclideanDistance(u: SparseVector, v: SparseVector): Double = {
    val uIndices = u.indices
    val vIndices = v.indices

    val uValues = u.values
    val vValues = v.values

    var sum = 0.0

    var i = 0
    var j = 0

    while(i < uIndices.length && j < vIndices.length) {
      if (uIndices(i) < vIndices(j)) {
        sum += pow(uValues(i), 2)
        i += 1
      } else if (uIndices(i) > vIndices(j)) {
        sum += pow(vValues(j), 2)
        j += 1
      } else {
        val dp = uValues(i) - vValues(j)
        sum += pow(dp, 2)
        i += 1
        j += 1
      }
    }

    while(i < uIndices.length) {
      sum += pow(uValues(i), 2)
      i += 1
    }

    while(j < vIndices.length) {
      sum += pow(vValues(j), 2)
      j += 1
    }

    sqrt(sum)
  }

  /**
   * Calculates the manhattan distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Manhattan distance between u and v.
   */
  def manhattanDistance(u: SparseVector, v: SparseVector): Double = {
    val uIndices = u.indices
    val vIndices = v.indices

    val uValues = u.values
    val vValues = v.values

    var sum = 0.0

    var i = 0
    var j = 0

    while(i < uIndices.length && j < vIndices.length) {
      if (uIndices(i) < vIndices(j)) {
        sum += abs(uValues(i))
        i += 1
      } else if (uIndices(i) > vIndices(j)) {
        sum += abs(vValues(j))
        j += 1
      } else {
        sum += abs(uValues(i) - vValues(j))
        i += 1
        j += 1
      }
    }

    while(i < uIndices.length) {
      sum += abs(uValues(i))
      i += 1
    }

    while(j < vIndices.length) {
      sum += abs(vValues(j))
      j += 1
    }

    sum
  }

  private def norm(u: SparseVector): Double = sqrt(u.values.map(v => v * v).sum)

  private def innerProduct(u: SparseVector, v: SparseVector): Double = {
    val uIndices = u.indices
    val vIndices = v.indices

    val uValues = u.values
    val vValues = v.values

    var dot = 0.0

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
