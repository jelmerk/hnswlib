package com.github.jelmerk.spark.conversion

import com.github.jelmerk.spark.linalg.Normalizer
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, FloatType, StructField, StructType}

/**
  * Companion class for VectorTransformer.
  */
object VectorConverter extends DefaultParamsReadable[Normalizer] {
  override def load(path: String): Normalizer = super.load(path)
}

/**
  * Converts the input vector to a float array.
  *
  * @param uid identifier
  */
class VectorConverter(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with Logging with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("trans"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.withColumn(getOutputCol, dataset.schema(getInputCol).dataType match {
      case dataType: DataType if dataType.typeName == "vector" => vectorToFloatArray(col(getInputCol))
      case ArrayType(DoubleType, _) => doubleArrayToFloatArray(col(getInputCol))
      case _ => throw new IllegalArgumentException("Not a valid input type")
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
      case dataType: DataType if dataType.typeName == "vector" => true
      case ArrayType(DoubleType, _) => true
      case _ => false
    }

    if (!inputColHasValidDataType) {
      throw new IllegalArgumentException(s"Input column $getInputCol must be a double array or vector.")
    }

    val outputFields = schema.fields :+ StructField(getOutputCol, ArrayType(FloatType), inputColumnSchema.nullable)
    StructType(outputFields)
  }


  private val vectorToFloatArray: UserDefinedFunction = udf { vector: Vector => vector.toArray.map(_.toFloat) }

  private val doubleArrayToFloatArray: UserDefinedFunction = udf { vector: Seq[Double] => vector.map(_.toFloat) }

}