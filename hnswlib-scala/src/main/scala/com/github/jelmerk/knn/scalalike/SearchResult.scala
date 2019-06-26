package com.github.jelmerk.knn.scalalike

import java.util.Comparator

object SearchResult {

  def apply[TItem, TDistance](item: TItem, distance: TDistance)(implicit ordering: Ordering[TDistance])
    : SearchResult[TItem, TDistance] =
      new SearchResult[TItem, TDistance](item, distance, ordering)

  def apply[TItem, TDistance](item: TItem, distance: TDistance, distanceComparator: Comparator[TDistance])
    : SearchResult[TItem, TDistance] =
      new SearchResult[TItem, TDistance](item, distance, distanceComparator)

  def unapply[TItem, TDistance](result: SearchResult[TItem, TDistance]): Option[(TItem, TDistance)] =
    Some(result.item -> result.distance)

}
