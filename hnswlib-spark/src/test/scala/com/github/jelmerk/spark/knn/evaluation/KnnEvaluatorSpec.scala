package com.github.jelmerk.spark.knn.evaluation

import com.github.jelmerk.spark.knn.hnsw.Neighbor
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FunSuite
import org.scalatest.Matchers._

class KnnEvaluatorSpec extends FunSuite with DataFrameSuiteBase {

  test("evaluate performance") {
    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val evaluator = new KnnEvaluator()
      .setApproximateNeighborsCol("approximate")
      .setExactNeighborsCol("exact")

    val df = sc.parallelize(Seq(
      Seq(Neighbor("1", 0.1f)) -> Seq(Neighbor("1", 0.1f)),
      Seq(Neighbor("2", 0.1f)) -> Seq(Neighbor("2", 0.1f), Neighbor("3", 0.9f))
    )).toDF("approximate", "exact")

    evaluator.evaluate(df) should be (0.75)

  }

}
