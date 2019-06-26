package com.github.jelmerk.knn.scalalike.statistics

object IndexStats {

  def apply(precision: Double): IndexStats = new IndexStats(precision)

  def unapply(stats: IndexStats): Option[Double] =
    Some(stats.precision())
}
