package com.github.jelmerk.knn.spark.ml.hnsw

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.FunSuite
import org.scalatest.Matchers._

case class VectorInputRow(id: Int, vector: Vector)

case class ArrayInputRow(id: String, vector: Array[Float])

class Person(val name: String) extends Serializable

class HnswSpec extends FunSuite with DatasetSuiteBase {

//  override def conf: SparkConf = super.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  test("vector input row") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val input = sc.parallelize(Seq(
      VectorInputRow(1, Vectors.dense(0.0110, 0.2341)),
      VectorInputRow(2, Vectors.dense(0.2300, 0.3891)),
      VectorInputRow(3, Vectors.dense(0.4300, 0.9891))
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


  test("array input row") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val input = sc.parallelize(Seq(
      ArrayInputRow("1", Array(0.0110f, 0.2341f)),
      ArrayInputRow("2", Array(0.2300f, 0.3891f))
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
