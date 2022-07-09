package com.github.jelmerk.spark.linalg

import com.github.jelmerk.knn.util.VectorUtils
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

/**
  * Companion class for Normalizer.
  */
object Normalizer extends DefaultParamsReadable[Normalizer] {
  override def load(path: String): Normalizer = super.load(path)
}

/**
  * Normalizes vectors to unit norm.
  *
  * @param uid identifier
  */
class Normalizer(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with Logging with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("norm"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = dataset.schema(getInputCol).dataType match {
    case VectorType => dataset.withColumn(getOutputCol, normalizeVector(col(getInputCol)))
    case ArrayType(FloatType, _) =>  dataset.withColumn(getOutputCol, normalizeFloatArray(col(getInputCol)))
    case ArrayType(DoubleType, _) => dataset.withColumn(getOutputCol, normalizeDoubleArray(col(getInputCol)))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains(getOutputCol)) {
      throw new IllegalArgumentException(s"Output column $getOutputCol already exists.")
    }

    if (!schema.fieldNames.contains(getInputCol)) {
      throw new IllegalArgumentException(s"Input column $getInputCol does not exist.")
    }

    val inputColumnSchema = schema(getInputCol)

    val inputColHasValidDataType = inputColumnSchema.dataType match {
      case VectorType => true
      case ArrayType(FloatType, _) => true
      case ArrayType(DoubleType, _) => true
      case _ => false
    }

    if (!inputColHasValidDataType) {
      throw new IllegalArgumentException(s"Input column $getInputCol must be a float array, double array or vector.")
    }

    schema
      .add(getOutputCol, inputColumnSchema.dataType, inputColumnSchema.nullable)
  }

  private def magnitude(vector: Vector): Double = {
    val values = vector match {
      case v: SparseVector => v.values
      case v: DenseVector => v.values
    }
    Math.sqrt(values.map(v => v * v).sum)
  }

  private val normalizeFloatArray: UserDefinedFunction = udf { value: Seq[Float] => VectorUtils.normalize(value.toArray) }

  private val normalizeDoubleArray: UserDefinedFunction = udf { value: Seq[Double] => VectorUtils.normalize(value.toArray) }

  private val normalizeVector: UserDefinedFunction = udf[Vector, Vector] { value =>
    val normFactor = 1 / magnitude(value)

    value match {
      case v: SparseVector => new SparseVector(v.size, v.indices, v.values.map(_ * normFactor))
      case v: DenseVector => new DenseVector(v.values.map(_ * normFactor))
    }
  }

}


