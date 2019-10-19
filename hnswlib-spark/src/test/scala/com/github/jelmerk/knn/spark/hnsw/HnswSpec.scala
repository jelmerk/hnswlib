package com.github.jelmerk.knn.spark.hnsw

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.Matchers._
import org.scalatest.prop.TableDrivenPropertyChecks._

case class InputRow[TId, TVector](id: TId, vector: TVector)

case class Neighbor[TId](neighbor: TId, distance: Float)

case class OutputRow[TId](id: TId, neighbors: Seq[Neighbor[TId]])

class HnswSpec extends FunSuite with DataFrameSuiteBase {

  // for some reason kryo cannot serialize the hnswindex so configure it to make sure it never gets serialized
  override def conf: SparkConf = super.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  test("find neighbors") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._


    val scenarios = Table[Boolean, DataFrame, DataFrame => Unit](
      ("excludeSelf", "input", "expectedOutput"),
      (false,
       sc.parallelize(Seq(
         InputRow(1000000, Vectors.dense(0.0110f, 0.2341f)),
         InputRow(2000000, Vectors.dense(0.2300f, 0.3891f)),
         InputRow(3000000, Vectors.dense(0.4300f, 0.9891f))
       )).toDF(),
       _.as[OutputRow[Int]].collect() should contain only (
         OutputRow(2000000, Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
         OutputRow(3000000, Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
         OutputRow(1000000, Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
       )
      ),
      (false,
       sc.parallelize(Seq(
         InputRow(1000000, Array(0.0110d, 0.2341d)),
         InputRow(2000000, Array(0.2300d, 0.3891d)),
         InputRow(3000000, Array(0.4300d, 0.9891d))
       )).toDF(),
       _.as[OutputRow[Int]].collect() should contain only (
         OutputRow(2000000, Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
         OutputRow(3000000, Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
         OutputRow(1000000, Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
       )
      ),
      (false,
       sc.parallelize(Seq(
         InputRow("1000000", Array(0.0110f, 0.2341f)),
         InputRow("2000000", Array(0.2300f, 0.3891f)),
         InputRow("3000000", Array(0.4300f, 0.9891f))
       )).toDF(),
       _.as[OutputRow[String]].collect() should contain only (
         OutputRow("2000000", Seq(Neighbor("2000000", 0.0f), Neighbor("3000000", 0.0076490045f), Neighbor("1000000", 0.11621308f))),
         OutputRow("3000000", Seq(Neighbor("3000000", 0.0f), Neighbor("2000000", 0.0076490045f), Neighbor("1000000", 0.06521261f))),
         OutputRow("1000000", Seq(Neighbor("1000000", 0.0f), Neighbor("3000000", 0.06521261f), Neighbor("2000000", 0.11621308f)))
       )
      ),
      (true,
       sc.parallelize(Seq(
         InputRow(1000000, Vectors.dense(0.0110f, 0.2341f)),
         InputRow(2000000, Vectors.dense(0.2300f, 0.3891f)),
         InputRow(3000000, Vectors.dense(0.4300f, 0.9891f))
       )).toDF(),
       _.as[OutputRow[Int]].collect() should contain only (
         OutputRow(2000000, Seq(Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
         OutputRow(3000000, Seq(Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
         OutputRow(1000000, Seq(Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
       )
      )
    )

    forAll (scenarios) { case (excludeSelf, input, validator) =>

      val hnsw = new Hnsw()
        .setIdentityCol("id")
        .setVectorCol("vector")
        .setNumPartitions(5)
        .setK(10)
        .setNeighborsCol("neighbors")
        .setExcludeSelf(excludeSelf)

      val model = hnsw.fit(input)

      val result = model.transform(input)

      validator(result)
    }
  }
}