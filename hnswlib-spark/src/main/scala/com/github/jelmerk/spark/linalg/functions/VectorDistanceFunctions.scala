package com.github.jelmerk.spark.linalg.functions

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

object VectorDistanceFunctions {

  /**
   * Calculates the cosine distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   * @return Cosine distance between u and v.
   */
  def cosineDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.cosineDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.cosineDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.cosineDistance(ud.toSparse, vs)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.cosineDistance(us, vd.toSparse)
  }

  /**
   * Calculates the inner product.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Cosine distance between u and v.
   */
  def innerProduct(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.innerProduct(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.innerProductDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.innerProductDistance(ud.toSparse, vs)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.innerProductDistance(us, vd.toSparse)
  }

  /**
   * Calculates the Bray Curtis distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Bray Curtis distance between u and v.
   */
  def brayCurtisDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.brayCurtisDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.brayCurtisDistance(us, vs)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.brayCurtisDistance(us, vd.toSparse)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.brayCurtisDistance(ud.toSparse, vs)
  }

  /**
   * Calculates the canberra distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Canberra distance between u and v.
   */
  def canberraDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.canberraDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.canberraDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.canberraDistance(ud.toSparse, vs)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.canberraDistance(us, vd.toSparse)
  }

  /**
   * Calculates the correlation distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Correlation distance between u and v.
   */
  def correlationDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.correlationDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.correlationDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => DenseVectorDistanceFunctions.correlationDistance(ud, vs.toDense)
    case (us: SparseVector, vd: DenseVector) => DenseVectorDistanceFunctions.correlationDistance(us.toDense, vd)
  }

  /**
   * Calculates the euclidean distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Euclidean distance between u and v.
   */
  def euclideanDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.euclideanDistance(ud, vd)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.euclideanDistance(us, vd.toSparse)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.euclideanDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.euclideanDistance(ud.toSparse, vs)
  }

  /**
   * Calculates the manhattan distance.
   *
   * @param u Left vector.
   * @param v Right vector.
   *
   * @return Manhattan distance between u and v.
   */
  def manhattanDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.manhattanDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.manhattanDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.manhattanDistance(ud.toSparse.toSparse, vs)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.manhattanDistance(us, vd.toSparse)
  }
}
