package com.github.jelmerk.spark.knn.hnsw

import java.io.File
import java.nio.file.Files
import java.util.UUID

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite
import org.scalatest.Matchers._
import org.scalatest.prop.TableDrivenPropertyChecks._

class SparseHnswSpec extends FunSuite with DataFrameSuiteBase {

  // for some reason kryo cannot serialize the hnswindex so configure it to make sure it never gets serialized
  override def conf: SparkConf = super.conf
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//    .set("spark.kryo.registrator", classOf[HnswLibKryoRegistrator].getName)

  
  test("find neighbors") {

    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val sparseVectorInput = sc.parallelize(Seq(
      InputRow(1000000, Vectors.sparse(2, Array(0, 1), Array(0.04693667884771435f,0.9988978667405123f))),
      InputRow(2000000, Vectors.sparse(2, Array(0, 1), Array(0.5088560419074649f, 0.8608516298493418f))),
      InputRow(3000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f)))
    )).toDF()

    val sparseVectorScenarioValidator: DataFrame => Unit = df =>
      df.as[FullOutputRow[Int, DenseVector]].collect() should contain only (
        FullOutputRow(2000000, Vectors.sparse(2, Array(0, 1), Array(0.5088560419074649f, 0.8608516298493418f)), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
        FullOutputRow(3000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f)), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        FullOutputRow(1000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f)), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val minimalSparseVectorScenarioValidator: DataFrame => Unit = df =>
      df.as[MinimalOutputRow[Int]].collect() should contain only (
        MinimalOutputRow(2000000, Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f), Neighbor(1000000, 0.11621308f))),
        MinimalOutputRow(3000000, Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        MinimalOutputRow(1000000, Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f), Neighbor(2000000, 0.11621308f)))
      )

    val similarityThresholdScenrioValidator: DataFrame => Unit = df =>
      df.as[FullOutputRow[Int, DenseVector]].collect() should contain only (
        FullOutputRow(2000000, Vectors.sparse(2, Array(0, 1), Array(0.5088560419074649f, 0.8608516298493418f)), Seq(Neighbor(2000000, 0.0f), Neighbor(3000000, 0.0076490045f))),
        FullOutputRow(3000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f)), Seq(Neighbor(3000000, 0.0f), Neighbor(2000000, 0.0076490045f), Neighbor(1000000, 0.06521261f))),
        FullOutputRow(1000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f)), Seq(Neighbor(1000000, 0.0f), Neighbor(3000000, 0.06521261f)))
      )


    val scenarios = Table[String, Boolean, Float, DataFrame, DataFrame => Unit](
      ("outputFormat", "excludeSelf", "similarityThreshold", "input",          "validator"),
      ("full",         false,         1f,                    sparseVectorInput, sparseVectorScenarioValidator),
      ("minimal",      false,         1f,                    sparseVectorInput, minimalSparseVectorScenarioValidator),
      ("full",         false,         0.1f,                  sparseVectorInput, similarityThresholdScenrioValidator)
    )

    forAll (scenarios) { case (outputFormat, excludeSelf, similarityThreshold, input, validator) =>

      val hnsw = new SparseHnsw()
        .setIdentifierCol("id")
        .setVectorCol("vector")
        .setDistanceFunction("inner-product")
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

    val hnsw = new SparseHnsw()
      .setIdentifierCol("id")
      .setVectorCol("vector")
      .setNeighborsCol("neighbors")
      .setOutputFormat("minimal")
      .setDistanceFunction("inner-product")

    val items = sc.parallelize(Seq(
      InputRow(1000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f))),
      InputRow(2000000, Vectors.sparse(2, Array(0, 1), Array(0.5088560419074649f, 0.8608516298493418f))),
      InputRow(3000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f)))
    )).toDF()

    withTempFolder { folder =>

      val path = new File(folder, "model").getCanonicalPath

      hnsw.fit(items).write.overwrite.save(path)

      val model = SparseHnswModel.load(path)

      val queryItems = sc.parallelize(Seq(
        InputRow(1000000, Vectors.sparse(2, Array(0, 1), Array(0.3986922200509975f, 0.9170847908840312f)))
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