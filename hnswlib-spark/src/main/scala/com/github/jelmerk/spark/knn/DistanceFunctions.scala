package com.github.jelmerk.spark.knn

import org.apache.spark.ml.linalg.{SparseVector, Vector}


object DistanceFunctions {

  def innerProduct(u: Vector, v: Vector): Double = {
    var dot = 0.0
    for (i <-  v.asInstanceOf[SparseVector].indices) {
      dot += u(i) * v(i)
    }
    1 - dot
  }
}
