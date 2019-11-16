package com.github.jelmerk.spark.linalg

import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

class Normalize(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with Logging {

  def this() = this(Identifiable.randomUID("norm"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = dataset.withColumn(getOutputCol, normalizeSparseUdf(col(getInputCol)))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), StructType(Seq(
        StructField("type", ByteType, nullable = false),
        StructField("size", IntegerType, nullable = true),
        StructField("indices", ArrayType(IntegerType, containsNull = false), nullable = true),
        StructField("values", ArrayType(DoubleType, containsNull = false), nullable = true))), nullable = false)
    StructType(outputFields)
  }

  private def magnitudeSparse(vector: SparseVector): Double = {
    var magnitude = 0.0
    for (aDouble <- vector.indices.map(vector.apply)) {
      magnitude += aDouble * aDouble
    }
    Math.sqrt(magnitude)
  }

  private def normalizeSparse(vector: SparseVector): SparseVector = {
    val normFactor = 1 / magnitudeSparse(vector)
    new SparseVector(vector.size, vector.indices, vector.values.map(_ * normFactor))
  }

  private val normalizeSparseUdf: UserDefinedFunction = udf { v: Vector => normalizeSparse(v.asInstanceOf[SparseVector]) }

}
