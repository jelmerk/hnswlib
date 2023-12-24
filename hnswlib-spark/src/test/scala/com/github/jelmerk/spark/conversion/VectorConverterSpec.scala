package com.github.jelmerk.spark.conversion

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.prop.TableDrivenPropertyChecks._

case class InputRow[TVector](vector: TVector)

case class OutputRow[TVectorIn, TVectorOut](vector: TVectorIn, array: TVectorOut)

class VectorConverterSpec extends AnyFunSuite with DataFrameSuiteBase {

  test("convert vectors") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val scenarios = Table[DataFrame, DataFrame, String](
      ("input", "expectedOutput", "outputType"), (
        Seq(InputRow(Vectors.dense(Array(1d, 2d, 3d)))).toDF(),
        Seq(OutputRow(Vectors.dense(Array(1d, 2d, 3d)), Array(1f, 2f, 3f))).toDF(),
        "array<float>"
      ), (
        Seq(InputRow(Array(1d, 2d, 3d))).toDF(),
        Seq(OutputRow(Array(1d, 2d, 3d), Array(1f, 2f, 3f))).toDF(),
        "array<float>"
      ), (
        Seq(InputRow(Vectors.dense(Array(1d, 2d, 3d)))).toDF(),
        Seq(OutputRow(Vectors.dense(Array(1d, 2d, 3d)), Array(1d, 2d, 3d))).toDF(),
        "array<double>"
      ), (
        Seq(InputRow(Array(1f, 2f, 3f))).toDF(),
        Seq(OutputRow(Array(1f, 2f, 3f), Array(1d, 2d, 3d))).toDF(),
        "array<double>"
      ), (
        Seq(InputRow(Array(1f, 2f, 3f))).toDF(),
        Seq(OutputRow(Array(1f, 2f, 3f), Vectors.dense(Array(1d, 2d, 3d)))).toDF(),
        "vector"
      ), (
        Seq(InputRow(Array(1d, 2d, 3d))).toDF(),
        Seq(OutputRow(Array(1d, 2d, 3d), Vectors.dense(Array(1d, 2d, 3d)))).toDF(),
        "vector"
      )
    )

    val input = Seq(InputRow(Array(1d, 2d, 3d))).toDF()
    val expectedOutput = Seq(OutputRow(Array(1d, 2d, 3d), Vectors.dense(Array(1d, 2d, 3d)))).toDF()
    val outputType = "vector"

//    forAll (scenarios) { case (input, expectedOutput, outputType) =>

      val converter = new VectorConverter()
        .setInputCol("vector")
        .setOutputCol("array")
        .setOutputType(outputType)

//      converter.transform(input).show()
      assertDataFrameEquals(converter.transform(input), expectedOutput)
//    }

  }
}
