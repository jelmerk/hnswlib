package com.github.jelmerk.spark.conversion

import com.github.jelmerk.spark.linalg.Normalizer
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, FloatType, StructType}

/**
  * Companion class for VectorConverter.
  */
object VectorConverter extends DefaultParamsReadable[Normalizer] {
  override def load(path: String): Normalizer = super.load(path)
}

private[conversion] trait VectorConverterParams extends HasInputCol with HasOutputCol {

  /**
    * Param for the type of vector to produce. one of array<float>, array<double>, vector
    * Default: "array<float>"
    *
    * @group param
    */
  final val outputType: Param[String] = new Param[String](this, "outputType", "type of vector to produce")

  /** @group getParam */
  final def getOutputType: String = $(outputType)

  setDefault(outputType -> "array<float>")
}

/**
  * Converts the input vector to a vector of another type.
  *
  * @param uid identifier
  */
class VectorConverter(override val uid: String)
  extends Transformer with VectorConverterParams with Logging with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("conv"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setOutputType(value: String): this.type = set(outputType, value)

  override def transform(dataset: Dataset[_]): DataFrame = {

    dataset.withColumn(getOutputCol, (dataset.schema(getInputCol).dataType, getOutputType) match {
      case (ArrayType(FloatType, _), "array<double>") => floatArrayToDoubleArray(col(getInputCol))
      case (ArrayType(FloatType, _), "vector") => floatArrayToVector(col(getInputCol))

      case (ArrayType(DoubleType, _), "array<float>") => doubleArrayToFloatArray(col(getInputCol))
      case (ArrayType(DoubleType, _), "vector") => doubleArrayToVector(col(getInputCol))

      case (VectorType, "array<float>") => vectorToFloatArray(col(getInputCol))
      case (VectorType, "array<double>") => vectorToDoubleArray(col(getInputCol))

      case _ => throw new IllegalArgumentException("Cannot convert vector")
    })
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
      case ArrayType(DoubleType, _) => true
      case _ => false
    }

    if (!inputColHasValidDataType) {
      throw new IllegalArgumentException(s"Input column $getInputCol must be a double array or vector.")
    }

    val outputType: DataType = getOutputType match {
      case "array<double>" => ArrayType(DoubleType)
      case "array<float>" => ArrayType(FloatType)
      case "vector" => VectorType
    }

    schema
      .add(getOutputCol, outputType, inputColumnSchema.nullable)
  }

  private val vectorToFloatArray: UserDefinedFunction = udf { vector: Vector => vector.toArray.map(_.toFloat) }

  private val doubleArrayToFloatArray: UserDefinedFunction = udf { vector: Seq[Double] => vector.map(_.toFloat) }

  private val floatArrayToDoubleArray: UserDefinedFunction = udf { vector: Seq[Float] => vector.toArray.map(_.toDouble) }

  private val vectorToDoubleArray: UserDefinedFunction = udf { vector: Vector => vector.toArray }

  private val floatArrayToVector: UserDefinedFunction = udf { vector: Seq[Float] => Vectors.dense(vector.map(_.toDouble).toArray) }

  private val doubleArrayToVector: UserDefinedFunction = udf { vector: Seq[Double] => Vectors.dense(vector.toArray) }

}