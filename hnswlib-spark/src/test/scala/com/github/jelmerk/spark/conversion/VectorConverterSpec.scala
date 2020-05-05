package com.github.jelmerk.spark.conversion

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.prop.TableDrivenPropertyChecks._

case class InputRow[TVector](vector: TVector)

case class OutputRow[TVector](vector: TVector, array: Array[Float])

class VectorConverterSpec extends FunSuite with DataFrameSuiteBase {

  test("convert vector to float vector") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val normalize = new VectorConverter()
      .setInputCol("vector")
      .setOutputCol("array")


    val scenarios = Table[DataFrame, DataFrame](
      ("input", "expectedOutput"),
      (
        Seq(InputRow(Vectors.dense(Array(0.01, 0.02, 0.03)))).toDF(), Seq(OutputRow(Vectors.dense(Array(0.01, 0.02, 0.03)),
                                                                                    Array(0.01f, 0.02f, 0.03f))).toDF()
      ), (
        Seq(InputRow(Array(0.01, 0.02, 0.03))).toDF(), Seq(OutputRow(Array(0.01, 0.02, 0.03),
                                                                     Array(0.01f, 0.02f, 0.03f))).toDF()
      )
    )

    forAll (scenarios) { case (input, expectedOutput) =>
      assertDataFrameEquals(normalize.transform(input), expectedOutput)
    }

  }
}
