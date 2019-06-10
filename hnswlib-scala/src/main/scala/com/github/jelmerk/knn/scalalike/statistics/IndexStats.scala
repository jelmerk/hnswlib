package com.github.jelmerk.knn.scalalike.statistics

object IndexStats {

  def unapply(stats: IndexStats): Option[Double] =
    Some(stats.precision())
}
