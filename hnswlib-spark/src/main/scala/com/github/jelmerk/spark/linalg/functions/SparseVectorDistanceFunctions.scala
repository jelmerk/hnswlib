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
   * Calculates inner product.
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
  def brayCurtisDistance(u: SparseVector, v: SparseVector): Double =
    throw new NotImplementedError("Not implemented for sparse vectors.")

  /**
   * Calculates the canberra distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Canberra distance between u and v.
   */
  def canberraDistance(u: SparseVector, v: SparseVector): Double =
    throw new NotImplementedError("Not implemented for sparse vectors.")

  /**
   * Calculates the correlation distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Correlation distance between u and v.
   */
  def correlationDistance(u: SparseVector, v: SparseVector): Double =
    throw new NotImplementedError("Not implemented for sparse vectors.")

  /**
   * Calculates the euclidean distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Euclidean distance between u and v.
   */
  def euclideanDistance(u: SparseVector, v: SparseVector): Double =
    throw new NotImplementedError("Not implemented for sparse vectors.")

  /**
   * Calculates the manhattan distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Manhattan distance between u and v.
   */
  def manhattanDistance(u: SparseVector, v: SparseVector): Double =
    throw new NotImplementedError("Not implemented for sparse vectors.")

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
