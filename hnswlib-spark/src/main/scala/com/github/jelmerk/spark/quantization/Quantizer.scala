package com.github.jelmerk.spark.quantization

import com.github.jelmerk.knn.util.VectorUtils
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Companion class for quantizer.
 */
object Quantizer extends DefaultParamsReadable[Quantizer] {
  override def load(path: String): Quantizer = super.load(path)
}
/**
 * Normalizes vectors to unit norm.
 *
 * @param uid identifier
 */
class Quantizer(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with Logging with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("quantize"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = dataset.schema(getInputCol).dataType match {
    case VectorType => dataset.withColumn(getOutputCol, quantizeVector(col(getInputCol)))
    case ArrayType(FloatType, _) => dataset.withColumn(getOutputCol, quantizeFloatArray(col(getInputCol)))
    case ArrayType(DoubleType, _) => dataset.withColumn(getOutputCol, quantizeDoubleArray(col(getInputCol)))
  }

  override def copy(extra: ParamMap): Quantizer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains(getOutputCol)) {
      throw new IllegalArgumentException(s"Output column $getOutputCol already exists.")
    }

    if (!schema.fieldNames.contains(getInputCol)) {
      throw new IllegalArgumentException(s"Input column $getInputCol does not exist.")
    }
    val inputColumnSchema = schema(getInputCol)

    schema
      .add(getOutputCol, ArrayType(ByteType), inputColumnSchema.nullable)
  }

  private val quantizeFloatArray: UserDefinedFunction = udf { value: Seq[Float] => VectorUtils.quantize(value.toArray) }

  private val quantizeDoubleArray: UserDefinedFunction = udf { value: Seq[Double] => VectorUtils.quantize(value.toArray) }

  private val quantizeVector: UserDefinedFunction = udf[Vector, Vector] { value =>
    ??? // TODO implement me
//    value match {
//      case v: SparseVector => new SparseVector(v.size, v.indices, v.values.map(_ * normFactor))
//      case v: DenseVector => new DenseVector(v.values.map(_ * normFactor))
//    }
  }

}
