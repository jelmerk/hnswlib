package com.github.jelmerk.knn.scalalike

object SearchResult {

  def unapply[TItem, TDistance](result: SearchResult[TItem, TDistance]): Option[(TItem, TDistance)] =
    Some(result.item -> result.distance)

}
