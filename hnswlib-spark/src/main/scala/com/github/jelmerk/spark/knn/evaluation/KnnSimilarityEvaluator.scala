package com.github.jelmerk.spark.knn.evaluation

import scala.reflect.runtime.universe._
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, IntegerType, LongType, StringType, StructField, StructType}

/**
  * Companion class for KnnSimilarityEvaluator.
  */
object KnnSimilarityEvaluator extends DefaultParamsReadable[KnnSimilarityEvaluator] {
  override def load(path: String): KnnSimilarityEvaluator = super.load(path)
}

/**
  * Evaluator for knn algorithms, which expects two input columns, the exact neighbors and approximate neighbors. It compares
  * the results to determine the accuracy of the approximate results. Typically you will want to compute this over a
  * small sample given the cost of computing the exact results on a large index.
  *
  * @param uid identifier
  */
class KnnSimilarityEvaluator(override val uid: String) extends Evaluator with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("knn_eval"))

  /**
    * Param for the column name for the approximate results.
    * Default: "approximateNeighbors"
    *
    * @group param
    */
  final val approximateNeighborsCol = new Param[String](this, "approximateNeighborsCol", "column containing the approximate neighbors")

  /**
    * @group getParam
    */
  final def getApproximateNeighborsCol: String = $(approximateNeighborsCol)

  /**
    * @group setParam
    */
  final def setApproximateNeighborsCol(value: String): this.type = set(approximateNeighborsCol, value)

  /**
    * Param for the column name for the exact results.
    * Default: "exactNeighbors"
    *
    * @group param
    */
  final val exactNeighborsCol = new Param[String](this, "exactNeighborsCol", "column containing the exact neighbors")

  /**
    * @group getParam
    */
  final def getExactNeighborsCol: String = $(exactNeighborsCol)

  /**
    * @group setParam
    */
  final def setExactNeighborsCol(value: String): this.type = set(exactNeighborsCol, value)

  /**
    * Returns the accuracy of the approximate results.
    *
    * @param dataset a dataset
    * @return the accuracy of the approximate results
    */
  override def evaluate(dataset: Dataset[_]): Double = {
    if (!dataset.schema.fieldNames.contains(getExactNeighborsCol)) throw new IllegalArgumentException(s"Column $getExactNeighborsCol does not exist.")
    if (!dataset.schema.fieldNames.contains(getApproximateNeighborsCol)) throw new IllegalArgumentException(s"Column $getApproximateNeighborsCol does not exist.")

    (dataset.schema(getExactNeighborsCol).dataType, dataset.schema(getApproximateNeighborsCol).dataType) match {
      case (ArrayType(StructType(Array(StructField("neighbor", IntegerType, _, _),
                                       StructField("distance", _, _, _))), _),
            ArrayType(StructType(Array(StructField("neighbor", IntegerType, _, _),
                                       StructField("distance", _, _, _))), _)) => typedEvaluate[Int](dataset)

      case (ArrayType(StructType(Array(StructField("neighbor", LongType, _, _),
                                       StructField("distance", _, _, _))), _),
            ArrayType(StructType(Array(StructField("neighbor", LongType, _, _),
                                       StructField("distance", _, _, _))), _)) => typedEvaluate[Long](dataset)

      case (ArrayType(StructType(Array(StructField("neighbor", StringType, _, _),
                                       StructField("distance", _, _, _))), _),
            ArrayType(StructType(Array(StructField("neighbor", StringType, _, _),
                                       StructField("distance", _, _, _))), _)) => typedEvaluate[String](dataset)

      case _ => throw new IllegalArgumentException(s"Column $getExactNeighborsCol and or $getApproximateNeighborsCol is not of the correct type.")
    }
  }

  private def typedEvaluate[TId : TypeTag](dataset: Dataset[_]): Double = {
    import dataset.sparkSession.implicits._

    dataset
      .select(
        col(s"$getExactNeighborsCol.neighbor"),
        col(s"$getApproximateNeighborsCol.neighbor")
      )
      .as[(Seq[TId], Seq[TId])]
      .mapPartitions( it => it.map { case (exactNeighbors, approximateNeighbors) =>
        exactNeighbors.toSet.intersect(approximateNeighbors.toSet).size -> exactNeighbors.size
      })
      .toDF("numMatching", "numResults")
      .select(when(sum($"numResults") === 0, 1.0).otherwise(sum($"numMatching") / sum($"numResults")))
      .as[Double]
      .collect()
      .head
  }

  override def copy(extra: ParamMap): Evaluator = this.defaultCopy(extra)

  override def isLargerBetter: Boolean = true

  setDefault(approximateNeighborsCol -> "approximateNeighbors", exactNeighborsCol -> "exactNeighbors")
}
