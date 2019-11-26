package com.github.jelmerk.spark.knn.hnsw

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.commons.lang.builder.{EqualsBuilder, HashCodeBuilder}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.Matchers._
import org.scalatest.prop.TableDrivenPropertyChecks._

case class InputRow[TId, TVector](id: TId, vector: TVector)

case class Neighbor[TId](neighbor: TId, distance: Float)

case class OutputRow[TId, TVector](id: TId, vector: TVector, neighbors: Seq[Neighbor[TId]]) {

  // case classes won't work because array equals is implemented as identity equality
  override def equals(other: Any): Boolean = EqualsBuilder.reflectionEquals(this, other)
  override def hashCode(): Int = HashCodeBuilder.reflectionHashCode(this)
}

class HnswSpec extends FunSuite with DataFrameSuiteBase {

  // for some reason kryo cannot serialize the hnswindex so configure it to make sure it never gets serialized
  override def conf: SparkConf = super.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

  test("find neighbors") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val noSimilarityThreshold = -1f

    val denseVectorInput = sc.parallelize(Seq(
      InputRow(1000000, Vectors.dense(0.0110f, 0.2341f)),
      InputRow(2000000, Vectors.dense(0.2300f, 0.3891f)),
      InputRow(3000000, Vectors.dense(0.4300f, 0.9891f))
    )).toDF()

    val denseVectorScenarioValidator: DataFrame => Unit = df =>
      df.as[OutputRow[Int, DenseVector]].collect() should contain only (
        OutputRow(2000000, Vectors.dense(0.2300f, 0.3891f), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
        OutputRow(3000000, Vectors.dense(0.4300f, 0.9891f), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        OutputRow(1000000, Vectors.dense(0.0110f, 0.2341f), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val similarityThresholdScenrioValidator: DataFrame => Unit = df =>
      df.as[OutputRow[Int, DenseVector]].collect() should contain only (
        OutputRow(2000000, Vectors.dense(0.2300f, 0.3891f), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f))),
        OutputRow(3000000, Vectors.dense(0.4300f, 0.9891f), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        OutputRow(1000000, Vectors.dense(0.0110f, 0.2341f), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f)))
      )

    val doubleArrayInput = sc.parallelize(Seq(
       InputRow(1000000, Array(0.0110d, 0.2341d)),
       InputRow(2000000, Array(0.2300d, 0.3891d)),
       InputRow(3000000, Array(0.4300d, 0.9891d))
     )).toDF()

    val doubleArrayScenarioValidator: DataFrame => Unit = df =>
      df.as[OutputRow[Int, Array[Double]]].collect() should contain only (
       OutputRow(2000000, Array(0.2300d, 0.3891d), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
       OutputRow(3000000, Array(0.4300d, 0.9891d), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
       OutputRow(1000000, Array(0.0110d, 0.2341d), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val floatArrayInput = sc.parallelize(Seq(
      InputRow("1000000", Array(0.0110f, 0.2341f)),
      InputRow("2000000", Array(0.2300f, 0.3891f)),
      InputRow("3000000", Array(0.4300f, 0.9891f))
    )).toDF()

    val floatArrayScenarioValidator: DataFrame => Unit = df =>
      df.as[OutputRow[String, Array[Float]]].collect() should contain only (
        OutputRow("2000000", Array(0.2300f, 0.3891f), Seq(Neighbor("2000000", 0.0f), Neighbor("3000000", 0.0076490045f), Neighbor("1000000", 0.11621308f))),
        OutputRow("3000000", Array(0.4300f, 0.9891f), Seq(Neighbor("3000000", 0.0f), Neighbor("2000000", 0.0076490045f), Neighbor("1000000", 0.06521261f))),
        OutputRow("1000000", Array(0.0110f, 0.2341f), Seq(Neighbor("1000000", 0.0f), Neighbor("3000000", 0.06521261f), Neighbor("2000000", 0.11621308f)))
      )

    val excludeSelfScenarioValidator: DataFrame => Unit = df =>
      df.as[OutputRow[Int, DenseVector]].collect() should contain only (
        OutputRow(2000000, Vectors.dense(0.2300f, 0.3891f), Seq(Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
        OutputRow(3000000, Vectors.dense(0.4300f, 0.9891f), Seq(Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        OutputRow(1000000, Vectors.dense(0.0110f, 0.2341f), Seq(Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val scenarios = Table[Boolean, Float, DataFrame, DataFrame => Unit](
      ("excludeSelf", "similarityThreshold", "input",          "validator"),
      (false,         1f,                    denseVectorInput, denseVectorScenarioValidator),
      (false,         0.1f,                  denseVectorInput, similarityThresholdScenrioValidator),
      (false,         noSimilarityThreshold, doubleArrayInput, doubleArrayScenarioValidator),
      (false,         noSimilarityThreshold, floatArrayInput,  floatArrayScenarioValidator),
      (true,          noSimilarityThreshold, denseVectorInput, excludeSelfScenarioValidator)
    )

    forAll (scenarios) { case (excludeSelf, similarityThreshold, input, validator) =>

      val hnsw = new Hnsw()
        .setIdentityCol("id")
        .setVectorCol("vector")
        .setNumPartitions(5)
        .setK(10)
        .setNeighborsCol("neighbors")
        .setExcludeSelf(excludeSelf)
        .setSimilarityThreshold(similarityThreshold)

      val model = hnsw.fit(input)

      val result = model.transform(input)

      validator(result)
    }
  }
}