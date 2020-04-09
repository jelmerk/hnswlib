package com.github.jelmerk.spark

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

package object functions {

  /**
    * Convert a dense vector to a float array.
    */
  val vectorToFloatArray: UserDefinedFunction = udf { vector: Vector => vector.toArray.map(_.toFloat) }

  /**
    * Convert a double array to a float array
    */
  val doubleArrayToFloatArray: UserDefinedFunction = udf { vector: Seq[Double] => vector.map(_.toFloat) }

}
