package com.github.jelmerk.spark.knn.hnsw

import java.io.File
import java.nio.file.Files
import java.util.UUID

import com.github.jelmerk.spark.HnswLibKryoRegistrator
import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.commons.io.FileUtils
import org.apache.commons.lang.builder.{EqualsBuilder, HashCodeBuilder}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.Matchers._
import org.scalatest.prop.TableDrivenPropertyChecks._



case class InputRow[TId, TVector](id: TId, vector: TVector)

case class Neighbor[TId](neighbor: TId, distance: Float)

case class FullOutputRow[TId, TVector](id: TId, vector: TVector, neighbors: Seq[Neighbor[TId]]) {

  // case classes won't work because array equals is implemented as identity equality
  override def equals(other: Any): Boolean = EqualsBuilder.reflectionEquals(this, other)
  override def hashCode(): Int = HashCodeBuilder.reflectionHashCode(this)
}

case class MinimalOutputRow[TId](id: TId, neighbors: Seq[Neighbor[TId]]) {

  // case classes won't work because array equals is implemented as identity equality
  override def equals(other: Any): Boolean = EqualsBuilder.reflectionEquals(this, other)
  override def hashCode(): Int = HashCodeBuilder.reflectionHashCode(this)
}

class HnswSpec extends FunSuite with DataFrameSuiteBase {

  // for some reason kryo cannot serialize the hnswindex so configure it to make sure it never gets serialized
  override def conf: SparkConf = super.conf
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//    .set("spark.kryo.registrator", classOf[HnswLibKryoRegistrator].getName)

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
      df.as[FullOutputRow[Int, DenseVector]].collect() should contain only (
        FullOutputRow(2000000, Vectors.dense(0.2300f, 0.3891f), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
        FullOutputRow(3000000, Vectors.dense(0.4300f, 0.9891f), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        FullOutputRow(1000000, Vectors.dense(0.0110f, 0.2341f), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val minimalDenseVectorScenarioValidator: DataFrame => Unit = df =>
      df.as[MinimalOutputRow[Int]].collect() should contain only (
        MinimalOutputRow(2000000, Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
        MinimalOutputRow(3000000, Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        MinimalOutputRow(1000000, Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val similarityThresholdScenrioValidator: DataFrame => Unit = df =>
      df.as[FullOutputRow[Int, DenseVector]].collect() should contain only (
        FullOutputRow(2000000, Vectors.dense(0.2300f, 0.3891f), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f))),
        FullOutputRow(3000000, Vectors.dense(0.4300f, 0.9891f), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        FullOutputRow(1000000, Vectors.dense(0.0110f, 0.2341f), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f)))
      )

    val doubleArrayInput = sc.parallelize(Seq(
       InputRow(1000000, Array(0.0110d, 0.2341d)),
       InputRow(2000000, Array(0.2300d, 0.3891d)),
       InputRow(3000000, Array(0.4300d, 0.9891d))
     )).toDF()

    val doubleArrayScenarioValidator: DataFrame => Unit = df =>
      df.as[FullOutputRow[Int, Array[Double]]].collect() should contain only (
       FullOutputRow(2000000, Array(0.2300d, 0.3891d), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
       FullOutputRow(3000000, Array(0.4300d, 0.9891d), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
       FullOutputRow(1000000, Array(0.0110d, 0.2341d), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val floatArrayInput = sc.parallelize(Seq(
      InputRow("1000000", Array(0.0110f, 0.2341f)),
      InputRow("2000000", Array(0.2300f, 0.3891f)),
      InputRow("3000000", Array(0.4300f, 0.9891f))
    )).toDF()

    val floatArrayScenarioValidator: DataFrame => Unit = df =>
      df.as[FullOutputRow[String, Array[Float]]].collect() should contain only (
        FullOutputRow("2000000", Array(0.2300f, 0.3891f), Seq(Neighbor("2000000", 0.0f), Neighbor("3000000", 0.0076490045f), Neighbor("1000000", 0.11621308f))),
        FullOutputRow("3000000", Array(0.4300f, 0.9891f), Seq(Neighbor("3000000", 0.0f), Neighbor("2000000", 0.0076490045f), Neighbor("1000000", 0.06521261f))),
        FullOutputRow("1000000", Array(0.0110f, 0.2341f), Seq(Neighbor("1000000", 0.0f), Neighbor("3000000", 0.06521261f), Neighbor("2000000", 0.11621308f)))
      )

    val excludeSelfScenarioValidator: DataFrame => Unit = df =>
      df.as[FullOutputRow[Int, DenseVector]].collect() should contain only (
        FullOutputRow(2000000, Vectors.dense(0.2300f, 0.3891f), Seq(Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
        FullOutputRow(3000000, Vectors.dense(0.4300f, 0.9891f), Seq(Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        FullOutputRow(1000000, Vectors.dense(0.0110f, 0.2341f), Seq(Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val scenarios = Table[String, Boolean, Float, DataFrame, DataFrame => Unit](
      ("outputFormat", "excludeSelf", "similarityThreshold", "input",          "validator"),
      ("full",         false,         1f,                    denseVectorInput, denseVectorScenarioValidator),
      ("minimal",      false,         1f,                    denseVectorInput, minimalDenseVectorScenarioValidator),
      ("full",         false,         0.1f,                  denseVectorInput, similarityThresholdScenrioValidator),
      ("full",         false,         noSimilarityThreshold, doubleArrayInput, doubleArrayScenarioValidator),
      ("full",         false,         noSimilarityThreshold, floatArrayInput,  floatArrayScenarioValidator),
      ("full",         true,          noSimilarityThreshold, denseVectorInput, excludeSelfScenarioValidator)
    )

    forAll (scenarios) { case (outputFormat, excludeSelf, similarityThreshold, input, validator) =>

      val hnsw = new Hnsw()
        .setIdentityCol("id")
        .setVectorCol("vector")
        .setNumPartitions(5)
        .setK(10)
        .setNeighborsCol("neighbors")
        .setExcludeSelf(excludeSelf)
        .setSimilarityThreshold(similarityThreshold)
        .setOutputFormat(outputFormat)

      val model = hnsw.fit(input)

      val result = model.transform(input)

      validator(result)
    }
  }

  test("save and load model") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val hnsw = new Hnsw()
      .setIdentityCol("id")
      .setVectorCol("vector")
      .setNeighborsCol("neighbors")
      .setOutputFormat("minimal")

    val items = sc.parallelize(Seq(
      InputRow(1000000, Vectors.dense(0.0110f, 0.2341f)),
      InputRow(2000000, Vectors.dense(0.2300f, 0.3891f)),
      InputRow(3000000, Vectors.dense(0.4300f, 0.9891f))
    )).toDF()

    withTempFolder { folder =>

      val path = new File(folder, "model").getCanonicalPath

      hnsw.fit(items).write.overwrite.save(path)

      val model = HnswModel.load(path)

      val queryItems = sc.parallelize(Seq(
        InputRow(1000000, Vectors.dense(0.0110f, 0.2341f))
      )).toDF()

      val results = model.transform(queryItems).as[MinimalOutputRow[Int]].collect()

      results.length should be(1)
      results.head should be (MinimalOutputRow(1000000, Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f))))
    }

  }

  def withTempFolder[T](fn: File => T): T = {
    val tempDir = Files.createTempDirectory(UUID.randomUUID().toString).toFile
    try {
      fn(tempDir)
    } finally {
      FileUtils.deleteDirectory(tempDir)
    }
  }


}