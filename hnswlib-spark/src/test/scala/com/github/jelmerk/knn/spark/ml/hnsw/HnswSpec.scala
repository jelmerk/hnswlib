package com.github.jelmerk.knn.spark.ml.hnsw

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.functions._
import org.scalatest.FunSuite
import org.scalatest.Matchers._

case class InputRow(id: String, vector: Vector)

class Person(val name: String) extends Serializable

class HnswSpec extends FunSuite with DatasetSuiteBase {


  test("knn search") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val input = sc.parallelize(Seq(
      InputRow("1", Vectors.dense(0.0110, 0.2341)),
      InputRow("2", Vectors.dense(0.2300, 0.3891))
    )).toDS

    val hnsw = new Hnsw()
      .setIdentityCol("id")
      .setVectorCol("vector")
      .setNumPartitions(5)
      .setK(10)
      .setNeighborsCol("neighbors")

    val model = hnsw.fit(input)

    val result = model.transform(input)

    result.printSchema()

    result.show()
  }


}
