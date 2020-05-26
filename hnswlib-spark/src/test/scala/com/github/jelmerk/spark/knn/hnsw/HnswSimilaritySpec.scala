package com.github.jelmerk.spark.knn.hnsw

import java.io.File
import java.nio.file.Files
import java.util.UUID

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.commons.io.FileUtils
import org.apache.commons.lang.builder.{EqualsBuilder, HashCodeBuilder}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.Matchers._
import org.scalatest.prop.TableDrivenPropertyChecks._

case class InputRow[TId, TVector](id: TId, vector: TVector)

case class Neighbor[TId, TDistance](neighbor: TId, distance: TDistance)

case class FullOutputRow[TId, TVector, TDistance](id: TId, vector: TVector, neighbors: Seq[Neighbor[TId, TDistance]]) {

  // case classes won't work because array equals is implemented as identity equality
  override def equals(other: Any): Boolean = EqualsBuilder.reflectionEquals(this, other)
  override def hashCode(): Int = HashCodeBuilder.reflectionHashCode(this)
}

case class MinimalOutputRow[TId, TDistance](id: TId, neighbors: Seq[Neighbor[TId, TDistance]]) {

  // case classes won't work because array equals is implemented as identity equality
  override def equals(other: Any): Boolean = EqualsBuilder.reflectionEquals(this, other)
  override def hashCode(): Int = HashCodeBuilder.reflectionHashCode(this)
}

class HnswSimilaritySpec extends FunSuite with DataFrameSuiteBase {

  // for some reason kryo cannot serialize the hnswindex so configure it to make sure it never gets serialized
  override def conf: SparkConf = super.conf
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//    .set("spark.kryo.registrator", classOf[HnswLibKryoRegistrator].getName)

  test("find neighbors") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val noSimilarityThreshold = -1.0

    val denseVectorInput = sc.parallelize(Seq(
      InputRow(1000000, Vectors.dense(0.0110, 0.2341)),
      InputRow(2000000, Vectors.dense(0.2300, 0.3891)),
      InputRow(3000000, Vectors.dense(0.4300, 0.9891))
    )).toDF()

    val denseVectorScenarioValidator: DataFrame => Unit = df => {
      val rows = df.as[FullOutputRow[Int, DenseVector, Double]].collect()

      rows.find(_.id == 1000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(1000000, 3000000, 2000000))
      rows.find(_.id == 2000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(2000000, 3000000, 1000000))
      rows.find(_.id == 3000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 2000000, 1000000))
    }

    val minimalDenseVectorScenarioValidator: DataFrame => Unit = df => {
      val rows = df.as[MinimalOutputRow[Int, Double]].collect()

      rows.find(_.id == 1000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(1000000, 3000000, 2000000))
      rows.find(_.id == 2000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(2000000, 3000000, 1000000))
      rows.find(_.id == 3000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 2000000, 1000000))
    }

    val sparseVectorInput = sc.parallelize(Seq(
      InputRow(1000000, Vectors.sparse(2, Array(0, 1), Array(0.0110, 0.2341))),
      InputRow(2000000, Vectors.sparse(2, Array(0, 1), Array(0.2300, 0.3891))),
      InputRow(3000000, Vectors.sparse(2, Array(0, 1), Array(0.4300, 0.9891)))
    )).toDF()

    val sparseVectorScenarioValidator: DataFrame => Unit = df => {
      val rows = df.as[FullOutputRow[Int, SparseVector, Double]].collect()

      rows.find(_.id == 1000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 2000000))
      rows.find(_.id == 2000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 1000000))
      rows.find(_.id == 3000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(2000000, 1000000))
    }

    val similarityThresholdScenarioValidator: DataFrame => Unit = df => {
      val rows = df.as[FullOutputRow[Int, DenseVector, Double]].collect()

      rows.find(_.id == 1000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(1000000, 3000000))
      rows.find(_.id == 2000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(2000000, 3000000))
      rows.find(_.id == 3000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 2000000, 1000000))
    }

    val doubleArrayInput = sc.parallelize(Seq(
       InputRow(1000000, Array(0.0110, 0.2341)),
       InputRow(2000000, Array(0.2300, 0.3891)),
       InputRow(3000000, Array(0.4300, 0.9891))
     )).toDF()

    val doubleArrayScenarioValidator: DataFrame => Unit = df => {
      val rows = df.as[FullOutputRow[Int, Array[Double], Double]].collect()

      rows.find(_.id == 1000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(1000000, 3000000, 2000000))
      rows.find(_.id == 2000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(2000000, 3000000, 1000000))
      rows.find(_.id == 3000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 2000000, 1000000))
    }

    val floatArrayInput = sc.parallelize(Seq(
      InputRow("1000000", Array(0.0110f, 0.2341f)),
      InputRow("2000000", Array(0.2300f, 0.3891f)),
      InputRow("3000000", Array(0.4300f, 0.9891f))
    )).toDF()

    val floatArrayScenarioValidator: DataFrame => Unit = df =>
      df.as[FullOutputRow[String, Array[Float], Float]].collect() should contain only (
        FullOutputRow("2000000", Array(0.2300f, 0.3891f), Seq(Neighbor("2000000", 0.0f), Neighbor("3000000", 0.0076490045f), Neighbor("1000000", 0.11621308f))),
        FullOutputRow("3000000", Array(0.4300f, 0.9891f), Seq(Neighbor("3000000", 0.0f), Neighbor("2000000", 0.0076490045f), Neighbor("1000000", 0.06521261f))),
        FullOutputRow("1000000", Array(0.0110f, 0.2341f), Seq(Neighbor("1000000", 0.0f), Neighbor("3000000", 0.06521261f), Neighbor("2000000", 0.11621308f)))
      )

    val excludeSelfScenarioValidator: DataFrame => Unit = df => {
      val rows = df.as[FullOutputRow[Int, SparseVector, Double]].collect()

      rows.find(_.id == 1000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 2000000))
      rows.find(_.id == 2000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(3000000, 1000000))
      rows.find(_.id == 3000000).toSeq.flatMap(_.neighbors.map(_.neighbor)) should be (Seq(2000000, 1000000))
    }

    val scenarios = Table[String, Boolean, Double, DataFrame, DataFrame => Unit](
      ("outputFormat", "excludeSelf", "similarityThreshold", "input",           "validator"),
      ("full",         false,         1,                     denseVectorInput,  denseVectorScenarioValidator),
      ("minimal",      false,         1,                     denseVectorInput,  minimalDenseVectorScenarioValidator),
      ("full",         false,         0.1,                   denseVectorInput,  similarityThresholdScenarioValidator),
      ("full",         false,         noSimilarityThreshold, doubleArrayInput,  doubleArrayScenarioValidator),
      ("full",         false,         noSimilarityThreshold, floatArrayInput,   floatArrayScenarioValidator),
      ("full",         true,          noSimilarityThreshold, denseVectorInput,  excludeSelfScenarioValidator),
      ("full",         true,          1,                     sparseVectorInput, sparseVectorScenarioValidator)
    )

    forAll (scenarios) { case (outputFormat, excludeSelf, similarityThreshold, input, validator) =>

      val hnsw = new HnswSimilarity()
        .setIdentifierCol("id")
        .setQueryIdentifierCol("id")
        .setFeaturesCol("vector")
        .setNumPartitions(5)
//        .setNumReplicas(3)
        .setK(10)
        .setExcludeSelf(excludeSelf)
        .setSimilarityThreshold(similarityThreshold)
        .setOutputFormat(outputFormat)

      val model = hnsw.fit(input).setPredictionCol("neighbors").setEf(10)

      val result = model.transform(input)

      validator(result)
    }
  }

  test("save and load model") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val hnsw = new HnswSimilarity()
      .setIdentifierCol("id")
      .setQueryIdentifierCol("id")
      .setFeaturesCol("vector")
      .setPredictionCol("neighbors")
      .setOutputFormat("minimal")

    val items = sc.parallelize(Seq(
      InputRow(1000000, Array(0.0110f, 0.2341f)),
      InputRow(2000000, Array(0.2300f, 0.3891f)),
      InputRow(3000000, Array(0.4300f, 0.9891f))
    )).toDF()

    withTempFolder { folder =>

      val path = new File(folder, "model").getCanonicalPath

      hnsw.fit(items).write.overwrite.save(path)

      val model = HnswSimilarityModel.load(path)

      val queryItems = sc.parallelize(Seq(
        InputRow(1000000, Array(0.0110f, 0.2341f))
      )).toDF()

      val results = model.transform(queryItems).as[MinimalOutputRow[Int, Float]].collect()

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