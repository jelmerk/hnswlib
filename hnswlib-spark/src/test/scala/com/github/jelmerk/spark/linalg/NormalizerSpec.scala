package com.github.jelmerk.spark.linalg

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.prop.TableDrivenPropertyChecks._

case class InputRow[TVector](vector: TVector)

case class OutputRow[TVector](vector: TVector, normalized: TVector)

class NormalizerSpec extends FunSuite with DataFrameSuiteBase {

  test("normalize vector") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val normalize = new Normalizer()
        .setInputCol("vector")
        .setOutputCol("normalized")

    val scenarios = Table[DataFrame, DataFrame](
      ("input", "expectedOutput"),
      (
        Seq(InputRow(new SparseVector(3, Array(0,1,2), Array(0.01, 0.02, 0.03)))).toDF(), Seq(OutputRow(new SparseVector(3, Array(0,1,2), Array(0.01, 0.02, 0.03)),
                                                                                                              new SparseVector(3, Array(0,1,2), Array(0.2672612419124244, 0.5345224838248488, 0.8017837257372731)))).toDF()
      ), (
        Seq(InputRow(new DenseVector(Array(0.01, 0.02, 0.03)))).toDF(), Seq(OutputRow(new DenseVector(Array(0.01, 0.02, 0.03)),
                                                                                      new DenseVector(Array(0.2672612419124244, 0.5345224838248488, 0.8017837257372731)))).toDF()
      ), (
        Seq(InputRow(Array(0.01, 0.02, 0.03))).toDF(), Seq(OutputRow(Array(0.01, 0.02, 0.03),
                                                                     Array(0.2672612419124244, 0.5345224838248488, 0.8017837257372731))).toDF()
      ), (
        Seq(InputRow(Array(0.01f, 0.02f, 0.03f))).toDF(), Seq(OutputRow(Array(0.01f, 0.02f, 0.03f), Array(0.26726124f, 0.5345225f, 0.8017837f))).toDF()
      )
    )

    forAll (scenarios) { case (input, expectedOutput) =>
      assertDataFrameEquals(normalize.transform(input), expectedOutput)
    }

  }
}
