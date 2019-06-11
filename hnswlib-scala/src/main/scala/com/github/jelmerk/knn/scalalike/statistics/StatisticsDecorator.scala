package com.github.jelmerk.knn.scalalike.statistics

import com.github.jelmerk.knn.scalalike.{Index, Item, JavaIndexAdapter, ScalaIndexAdapter}
import com.github.jelmerk.knn.statistics.{StatisticsDecorator => JStatisticsDecorator}

object StatisticsDecorator {

  /**
    * Decorator on top of an index that will collect statistics about the index. Such as the precision of the results
    * returned by the approximative index compared to a brute force baseline.
    *
    * @param delegate the approximative index
    * @param groundTruth the brute force index
    * @param maxPrecisionSampleFrequency at most maxPrecisionSampleFrequency the results from the approximative index
    *                                    will be compared with those of the groundTruth to establish the the runtime
    *                                    precision of the index.
    * @param numSamples number of samples to calculate the moving average over
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return the StatisticsDecorator
    */
  def apply[TId, TVector, TItem <: Item[TId, TVector], TDistance ]
    (delegate: Index[TId, TVector, TItem, TDistance],
     groundTruth: Index[TId, TVector, TItem, TDistance],
     maxPrecisionSampleFrequency: Int,
     numSamples: Int = JStatisticsDecorator.DEFAULT_NUM_SAMPLES)
      : StatisticsDecorator[TId, TVector, TItem, TDistance] = {

    new StatisticsDecorator(new JStatisticsDecorator(new JavaIndexAdapter(delegate),
      new JavaIndexAdapter(groundTruth), maxPrecisionSampleFrequency, numSamples))
  }
}

@SerialVersionUID(1L)
class StatisticsDecorator[TId, TVector, TItem <: Item[TId, TVector], TDistance](
                                                                                 delegate: JStatisticsDecorator[TId, TVector, TItem, TDistance])
  extends ScalaIndexAdapter[TId, TVector, TItem ,TDistance](delegate) {

  def stats: IndexStats = delegate.stats

}