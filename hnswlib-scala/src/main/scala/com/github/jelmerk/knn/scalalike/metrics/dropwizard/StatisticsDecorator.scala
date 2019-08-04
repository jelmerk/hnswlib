package com.github.jelmerk.knn.scalalike.metrics.dropwizard

import com.codahale.metrics._
import com.github.jelmerk.knn.scalalike.{Index, Item, JavaIndexAdapter, ScalaIndexAdapter}
import com.github.jelmerk.knn.metrics.dropwizard.{StatisticsDecorator => JStatisticsDecorator}

object StatisticsDecorator {

  /**
    * Decorator on top of an index that will collect statistics about the index. Such as the precision of the results
    * returned by the approximative index compared to a brute force baseline.
    *
    * @param approximativeIndex the approximative index
    * @param groundTruthIndex the brute force index
    * @param maxAccuracySampleFrequency at most every maxAccuracySampleFrequency requests compare the results of the
    *                                   approximate index with those of the ground truth index to establish the runtime
    *                                   accuracy of the index
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return the StatisticsDecorator
    */
  def apply[TId, TVector, TItem <: Item[TId, TVector], TDistance,
              TApproximativeIndex <: Index[TId, TVector, TItem, TDistance],
              TGroundTruthIndex <: Index[TId, TVector, TItem, TDistance]](
        indexName: String,
        approximativeIndex: TApproximativeIndex,
        groundTruthIndex: TGroundTruthIndex,
        maxAccuracySampleFrequency: Int)
      : StatisticsDecorator[TId, TVector, TItem, TDistance, TApproximativeIndex, TGroundTruthIndex] = {

    val javaApproximativeIndex = new JavaIndexAdapter(approximativeIndex)
    val javaGroundTruthIndex = new JavaIndexAdapter(groundTruthIndex)
    val javaStatisticsDecorator = new JStatisticsDecorator[TId, TVector, TItem, TDistance,
        JavaIndexAdapter[TId, TVector, TItem, TDistance], JavaIndexAdapter[TId, TVector, TItem, TDistance]](
      indexName, javaApproximativeIndex, javaGroundTruthIndex, maxAccuracySampleFrequency)

    new StatisticsDecorator(javaStatisticsDecorator)
  }


  /**
    * Decorator on top of an index that will collect statistics about the index. Such as the precision of the results
    * returned by the approximative index compared to a brute force baseline.
    *
    * @param approximativeIndex the approximative index
    * @param groundTruthIndex the brute force index
    * @param maxAccuracySampleFrequency at most every maxAccuracySampleFrequency requests compare the results of the
    *                                   approximate index with those of the ground truth index to establish the runtime
    *                                   accuracy of the index
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return the StatisticsDecorator
    */
  def apply[TId, TVector, TItem <: Item[TId, TVector], TDistance,
              TApproximativeIndex <: Index[TId, TVector, TItem, TDistance],
              TGroundTruthIndex <: Index[TId, TVector, TItem, TDistance]](
        metricRegistry: MetricRegistry,
        clazz: Class[_],
        indexName: String,
        approximativeIndex: TApproximativeIndex,
        groundTruthIndex: TGroundTruthIndex,
        maxAccuracySampleFrequency: Int)
      : StatisticsDecorator[TId, TVector, TItem, TDistance, TApproximativeIndex, TGroundTruthIndex] = {

    val javaApproximativeIndex = new JavaIndexAdapter(approximativeIndex)
    val javaGroundTruthIndex = new JavaIndexAdapter(groundTruthIndex)
    val javaStatisticsDecorator = new JStatisticsDecorator[TId, TVector, TItem, TDistance,
        JavaIndexAdapter[TId, TVector, TItem, TDistance], JavaIndexAdapter[TId, TVector, TItem, TDistance]](
      metricRegistry, clazz, indexName, javaApproximativeIndex, javaGroundTruthIndex, maxAccuracySampleFrequency)

    new StatisticsDecorator(javaStatisticsDecorator)
  }

}

/**

  * Decorator on top of an index that will collect statistics about the index. Such as the precision of the results
  * returned by the approximative index compared to a brute force baseline.
  *
  * @param delegate java class this adapter class delegates to
  * @tparam TId Type of the external identifier of an item
  * @tparam TVector Type of the vector to perform distance calculation on
  * @tparam TItem Type of items stored in the index
  * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
  * @tparam TApproximativeIndex Type of the approximative index
  * @tparam TGroundTruthIndex Type of the ground truth index
  */
@SerialVersionUID(1L)
class StatisticsDecorator[TId, TVector, TItem <: Item[TId, TVector], TDistance,
    TApproximativeIndex <: Index[TId, TVector, TItem, TDistance],
    TGroundTruthIndex <: Index[TId, TVector, TItem, TDistance]] private
      (delegate: JStatisticsDecorator[TId, TVector, TItem, TDistance, JavaIndexAdapter[TId, TVector, TItem, TDistance], JavaIndexAdapter[TId, TVector, TItem, TDistance]])
        extends ScalaIndexAdapter[TId, TVector, TItem ,TDistance](delegate) {

  /**
    * Returns the approximative index.
    *
    * @return the approximative index
    */
  def approximativeIndex: TApproximativeIndex =
    delegate.getApproximativeIndex.delegate.asInstanceOf[TApproximativeIndex]

  /**
    * Returns the groundtruth index.
    *
    * @return the groundtruth index
    */
  def groundTruthIndex: TGroundTruthIndex =
    delegate.getGroundTruthIndex.delegate.asInstanceOf[TGroundTruthIndex]

}