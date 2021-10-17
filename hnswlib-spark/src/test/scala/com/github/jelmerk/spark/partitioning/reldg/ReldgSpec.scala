package com.github.jelmerk.spark.partitioning.reldg

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.scalatest.FunSuite

class ReldgSpec extends FunSuite with DataFrameSuiteBase {

  test("partition data") {
    val sqlCtx = sqlContext
    import sqlCtx.implicits._

    val reldg = new Reldg()
      .setFeaturesCol("vector")
      .setPartitionCol("partition")
      .setQueryPartitionsCol("partitions")
      .setNumPartitions(2)
      .setDistanceFunction("euclidean")
      .setK(10)

    val centroids = Seq(
      Centroid(Vectors.dense(0.0110, 0.2341)),
      Centroid(Vectors.dense(0.2300, 0.3891)),
      Centroid(Vectors.dense(0.2430, 0.2891))
    ).toDF

    val model = reldg.fit(centroids)

    val data = Seq(
      Data(Vectors.dense(0.0110, 0.2341)),
      Data(Vectors.dense(0.1110, 0.3341)),
      Data(Vectors.dense(0.3110, 0.1341))
    ).toDF


    // todo fix this
    val result = model.transform(data)

    result.show()

  }

}


case class Centroid(vector: Vector)

case class Data(vector: Vector)