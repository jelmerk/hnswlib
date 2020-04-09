package com.github.jelmerk.spark.linalg

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.prop.TableDrivenPropertyChecks._

case class InputRow[TVector](vector: TVector)

case class OutputRow[TVector](vector: TVector, normalized: TVector)



class NormalizerSpec extends FunSuite with DataFrameSuiteBase {

  test("foo") {
    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val normalize = new Normalizer()
      .setInputCol("vector")
      .setOutputCol("normalized")

    val x = Seq(
      InputRow(Vectors.sparse(2, Array(0, 1), Array(0.0110f, 0.2341f))),
      InputRow(Vectors.sparse(2, Array(0, 1), Array(0.2300f, 0.3891f))),
      InputRow(Vectors.sparse(2, Array(0, 1), Array(0.4300f, 0.9891f)))
    ).toDS()

    normalize.transform(x).show(false)

  }

  test("normalize vector") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val normalize = new Normalizer()
        .setInputCol("vector")
        .setOutputCol("normalized")

    val scenarios = Table[DataFrame, DataFrame](
      ("input", "expectedOutput"),
      (
        sc.parallelize(Seq(InputRow(new SparseVector(3, Array(0,1,2), Array(0.01, 0.02, 0.03))))).toDF(),
        sc.parallelize(Seq(OutputRow(new SparseVector(3, Array(0,1,2), Array(0.01, 0.02, 0.03)),
                                     new SparseVector(3, Array(0,1,2), Array(0.2672612419124244, 0.5345224838248488, 0.8017837257372731))))).toDF()
      ),
      (
        sc.parallelize(Seq(InputRow(new DenseVector(Array(0.01, 0.02, 0.03))))).toDF(),
        sc.parallelize(Seq(OutputRow(new DenseVector(Array(0.01, 0.02, 0.03)),
          new DenseVector(Array(0.2672612419124244, 0.5345224838248488, 0.8017837257372731))))).toDF()
      ),
      (
        sc.parallelize(Seq(InputRow(Array(0.01, 0.02, 0.03)))).toDF(),
        sc.parallelize(Seq(OutputRow(Array(0.01, 0.02, 0.03), Array(0.2672612419124244, 0.5345224838248488, 0.8017837257372731)))).toDF()
      ),
      (
        sc.parallelize(Seq(InputRow(Array(0.01f, 0.02f, 0.03f)))).toDF(),
        sc.parallelize(Seq(OutputRow(Array(0.01f, 0.02f, 0.03f), Array(0.26726124f, 0.5345225f, 0.8017837f)))).toDF()
      )
    )

    forAll (scenarios) { case (input, expectedOutput) =>
      assertDataFrameEquals(normalize.transform(input), expectedOutput)
    }

  }
}
