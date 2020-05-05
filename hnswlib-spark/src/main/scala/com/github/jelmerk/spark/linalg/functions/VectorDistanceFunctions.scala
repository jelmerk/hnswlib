package com.github.jelmerk.spark.linalg.functions

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

object VectorDistanceFunctions {

  def cosineDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.cosineDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.cosineDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.cosineDistance(ud.toSparse, vs)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.cosineDistance(us, vd.toSparse)
  }

  def innerProduct(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.innerProduct(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.innerProductDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => SparseVectorDistanceFunctions.innerProductDistance(ud.toSparse, vs)
    case (us: SparseVector, vd: DenseVector) => SparseVectorDistanceFunctions.innerProductDistance(us, vd.toSparse)
  }

  def brayCurtisDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.brayCurtisDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.brayCurtisDistance(us, vs)
    case (us: SparseVector, vd: DenseVector) => DenseVectorDistanceFunctions.brayCurtisDistance(us.toDense, vd)
    case (ud: DenseVector, vs: SparseVector) => DenseVectorDistanceFunctions.brayCurtisDistance(ud, vs.toDense)
  }

  def canberraDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.canberraDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.canberraDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => DenseVectorDistanceFunctions.canberraDistance(ud, vs.toDense)
    case (us: SparseVector, vd: DenseVector) => DenseVectorDistanceFunctions.canberraDistance(us.toDense, vd)
  }

  def correlationDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.correlationDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.correlationDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => DenseVectorDistanceFunctions.correlationDistance(ud, vs.toDense)
    case (us: SparseVector, vd: DenseVector) => DenseVectorDistanceFunctions.correlationDistance(us.toDense, vd)
  }

  def euclideanDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.euclideanDistance(ud, vd)
    case (us: SparseVector, vd: DenseVector) => DenseVectorDistanceFunctions.euclideanDistance(us.toDense, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.euclideanDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => DenseVectorDistanceFunctions.euclideanDistance(ud, vs.toDense)
  }

  def manhattanDistance(u: Vector, v: Vector): Double = (u, v) match {
    case (ud: DenseVector, vd: DenseVector) => DenseVectorDistanceFunctions.manhattanDistance(ud, vd)
    case (us: SparseVector, vs: SparseVector) => SparseVectorDistanceFunctions.manhattanDistance(us, vs)
    case (ud: DenseVector, vs: SparseVector) => DenseVectorDistanceFunctions.manhattanDistance(ud, vs.toDense)
    case (us: SparseVector, vd: DenseVector) => DenseVectorDistanceFunctions.manhattanDistance(us.toDense, vd)
  }
}
