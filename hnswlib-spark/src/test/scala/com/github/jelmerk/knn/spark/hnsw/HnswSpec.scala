package com.github.jelmerk.knn.spark.hnsw

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.{FunSuite, OptionValues}
import org.scalatest.Matchers._

case class VectorInputRow(id: Int, vector: Vector)
case class VectorOutputRowNeighbor(neighbor: Int, distance: Float)
case class VectorOutputRow(id: Int, neighbors: Seq[VectorOutputRowNeighbor])

case class ArrayInputRowFloat(id: String, vector: Array[Float])
case class ArrayInputRowDouble(id: String, vector: Array[Double])
case class ArrayOutputRowNeighbor(neighbor: String, distance: Float)
case class ArrayOutputRow(id: String, neighbors: Seq[ArrayOutputRowNeighbor])

class HnswSpec extends FunSuite with DatasetSuiteBase {

  // for some reason kryo cannot serialize the hnswindex so configure it to make sure it never gets serialized
  override def conf: SparkConf = super.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  test("vector input row float") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val input = sc.parallelize(Seq(
      VectorInputRow(1000000, Vectors.dense(0.0110f, 0.2341f)),
      VectorInputRow(2000000, Vectors.dense(0.2300f, 0.3891f)),
      VectorInputRow(3000000, Vectors.dense(0.4300f, 0.9891f))
    )).toDS

    val hnsw = new Hnsw()
      .setIdentityCol("id")
      .setVectorCol("vector")
      .setNumPartitions(5)
      .setK(10)
      .setNeighborsCol("neighbors")

    val model = hnsw.fit(input)

    val result = model.transform(input).as[VectorOutputRow]
      .collect()

    result should have size 3
    result should contain(VectorOutputRow(2000000, Seq(VectorOutputRowNeighbor(3000000, 0.0076490045f), VectorOutputRowNeighbor(1000000, 0.11621308f))))
    result should contain(VectorOutputRow(3000000, Seq(VectorOutputRowNeighbor(2000000, 0.0076490045f), VectorOutputRowNeighbor(1000000, 0.06521261f))))
    result should contain(VectorOutputRow(1000000, Seq(VectorOutputRowNeighbor(3000000, 0.06521261f), VectorOutputRowNeighbor(2000000, 0.11621308f))))
  }

  test("array input row float") {
    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val input = sc.parallelize(Seq(
      ArrayInputRowFloat("1", Array(0.0110f, 0.2341f)),
      ArrayInputRowFloat("2", Array(0.2300f, 0.3891f)),
      ArrayInputRowFloat("3", Array(0.4300f, 0.9891f))
    )).toDS

    val hnsw = new Hnsw()
      .setIdentityCol("id")
      .setVectorCol("vector")
      .setNumPartitions(5)
      .setK(10)
      .setNeighborsCol("neighbors")

    val model = hnsw.fit(input)

    val result = model.transform(input).as[ArrayOutputRow]
      .collect()

    result should have size 3
    result should contain(ArrayOutputRow("2", Seq(ArrayOutputRowNeighbor("3", 0.0076490045f), ArrayOutputRowNeighbor("1", 0.11621308f))))
    result should contain(ArrayOutputRow("3", Seq(ArrayOutputRowNeighbor("2", 0.0076490045f), ArrayOutputRowNeighbor("1", 0.06521261f))))
    result should contain(ArrayOutputRow("1", Seq(ArrayOutputRowNeighbor("3", 0.06521261f), ArrayOutputRowNeighbor("2", 0.11621308f))))
  }

  test("vector input row double") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val input = sc.parallelize(Seq(
      VectorInputRow(1, Vectors.dense(0.0110d, 0.2341d)),
      VectorInputRow(2, Vectors.dense(0.2300d, 0.3891d)),
      VectorInputRow(3, Vectors.dense(0.4300d, 0.9891d))
    )).toDS

    val hnsw = new Hnsw()
      .setIdentityCol("id")
      .setVectorCol("vector")
      .setNumPartitions(5)
      .setK(10)
      .setNeighborsCol("neighbors")

    val model = hnsw.fit(input)

    val result = model.transform(input).as[VectorOutputRow]
      .collect()

    result should have size 3
    result should contain(VectorOutputRow(2, Seq(VectorOutputRowNeighbor(3, 0.0076490045f), VectorOutputRowNeighbor(1, 0.11621308f))))
    result should contain(VectorOutputRow(3, Seq(VectorOutputRowNeighbor(2, 0.0076490045f), VectorOutputRowNeighbor(1, 0.06521261f))))
    result should contain(VectorOutputRow(1, Seq(VectorOutputRowNeighbor(3, 0.06521261f), VectorOutputRowNeighbor(2, 0.11621308f))))
  }

  test("array input row double") {
    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val input = sc.parallelize(Seq(
      ArrayInputRowDouble("1", Array(0.0110d, 0.2341d)),
      ArrayInputRowDouble("2", Array(0.2300d, 0.3891d)),
      ArrayInputRowDouble("3", Array(0.4300d, 0.9891d))
    )).toDS

    val hnsw = new Hnsw()
      .setIdentityCol("id")
      .setVectorCol("vector")
      .setNumPartitions(5)
      .setK(10)
      .setNeighborsCol("neighbors")

    val model = hnsw.fit(input)

    val result = model.transform(input).as[ArrayOutputRow]
      .collect()

    result should have size 3
    result should contain(ArrayOutputRow("2", Seq(ArrayOutputRowNeighbor("3", 0.0076490045f), ArrayOutputRowNeighbor("1", 0.11621308f))))
    result should contain(ArrayOutputRow("3", Seq(ArrayOutputRowNeighbor("2", 0.0076490045f), ArrayOutputRowNeighbor("1", 0.06521261f))))
    result should contain(ArrayOutputRow("1", Seq(ArrayOutputRowNeighbor("3", 0.06521261f), ArrayOutputRowNeighbor("2", 0.11621308f))))
  }
}
