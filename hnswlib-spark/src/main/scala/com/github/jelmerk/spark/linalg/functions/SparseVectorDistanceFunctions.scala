package com.github.jelmerk.spark.linalg.functions

import org.apache.spark.ml.linalg.SparseVector

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

    if (denom == 0d) 1d
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

    var sump = 0d
    var sumn = 0d

    var i = 0
    var j = 0

    while(i < uIndices.length || j < vIndices.length) {
      if (j == vIndices.length || i < uIndices.length && uIndices(i) < vIndices(j)) {
        sumn += math.abs(uValues(i))
        sump += math.abs(uValues(i))
        i += 1
      } else if (i == uIndices.length || j < vIndices.length && uIndices(i) > vIndices(j)) {
        sumn += math.abs(vValues(j))
        sump += math.abs(vValues(j))
        j += 1
      } else {
        sumn += math.abs(uValues(i) - vValues(j))
        sump += math.abs(uValues(i) + vValues(j))
        i += 1
        j += 1
      }
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

    var distance = 0d

    var i = 0
    var j = 0

    while(i < uIndices.length || j < vIndices.length) {
      if (j == vIndices.length || i < uIndices.length && uIndices(i) < vIndices(j)) {
        distance += 1
        i += 1
      } else if (i == uIndices.length || j < vIndices.length && uIndices(i) > vIndices(j)) {
        distance += 1
        j += 1
      } else {
        distance += math.abs(uValues(i) - vValues(j)) / (math.abs(uValues(i)) + math.abs(vValues(j)))
        i += 1
        j += 1
      }
    }
    distance
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

    val absXSquared = math.abs(math.pow(x, 2))

    while(i < uIndices.length || j < vIndices.length) {
      if (j == vIndices.length || i < uIndices.length && uIndices(i) < vIndices(j)) {

        num += (uValues(i) + x) * y

        den1 += math.abs(math.pow(uValues(i) + x, 2))
        den2 += absXSquared
        left -= 1

        i += 1
      } else if (i == uIndices.length || j < vIndices.length && uIndices(i) > vIndices(j)) {

        num += x * (vValues(j) + y)

        den1 += absXSquared
        den2 += math.abs(math.pow(vValues(j) + x, 2))
        left -= 1
        j += 1
      } else {
        num += (uValues(i) + x) * (vValues(j) + y)

        den1 += math.abs(math.pow(uValues(i) + x, 2))
        den2 += math.abs(math.pow(vValues(j) + x, 2))
        left -= 1

        i += 1
        j += 1
      }
    }

    num += (x * y) * left
    den1 += absXSquared * left
    den2 += absXSquared * left

    1 - (num / (math.sqrt(den1) * math.sqrt(den2)))
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

    var sum = 0d

    var i = 0
    var j = 0

    while(i < uIndices.length || j < vIndices.length) {
      if (j == vIndices.length || i < uIndices.length && uIndices(i) < vIndices(j)) {
        sum += uValues(i) * uValues(i)
        i += 1
      } else if (i == uIndices.length || j < vIndices.length && uIndices(i) > vIndices(j)) {
        sum += vValues(j) * vValues(j)
        j += 1
      } else {
        val dp = uValues(i) - vValues(j)
        sum += dp * dp
        i += 1
        j += 1
      }
    }

    math.sqrt(sum)
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

    var sum = 0d

    var i = 0
    var j = 0

    while(i < uIndices.length || j < vIndices.length) {
      if (j == vIndices.length || i < uIndices.length && uIndices(i) < vIndices(j)) {
        sum += math.abs(uValues(i))
        i += 1
      } else if (i == uIndices.length || j < vIndices.length && uIndices(i) > vIndices(j)) {
        sum += math.abs(vValues(j))
        j += 1
      } else {
        sum += math.abs(uValues(i) - vValues(j))
        i += 1
        j += 1
      }
    }
    sum
  }

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
