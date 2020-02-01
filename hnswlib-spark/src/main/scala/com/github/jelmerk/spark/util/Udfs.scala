package com.github.jelmerk.spark.util

import com.github.jelmerk.knn.scalalike.{SparseVector => HnswLibSparseVector}

import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

private[spark] object Udfs {

  /**
   * Convert a dense vector to a float array.
   */
  val vectorToFloatArray: UserDefinedFunction = udf { vector: Vector => vector.toArray.map(_.toFloat) }

  /**
   * Convert a double array to a float array
   */
  val doubleArrayToFloatArray: UserDefinedFunction = udf { vector: Seq[Double] => vector.map(_.toFloat) }

  /**
   * Convert a mllib sparse vector to a hnswlib sparse vector.
   */
  val sparseVectorToHnswLibSparseVector: UserDefinedFunction = udf { vector: SparseVector =>
    new HnswLibSparseVector[Array[Float]](vector.indices, vector.values.map(_.toFloat))
  }
}
