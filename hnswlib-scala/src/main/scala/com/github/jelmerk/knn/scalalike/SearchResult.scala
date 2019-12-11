package com.github.jelmerk.knn.scalalike

import java.util.Comparator

object SearchResult {

  /**
    * SearchResult factory.
    *
    * @param item the item
    * @param distance the distance from the search query
    * @param ordering the ordering
    * @tparam TItem type of the item returned
    * @tparam TDistance type of the distance returned by the configured distance function
    * @return a new SearchResult
    */
  def apply[TItem, TDistance](item: TItem, distance: TDistance)(implicit ordering: Ordering[TDistance])
    : SearchResult[TItem, TDistance] =
      new SearchResult[TItem, TDistance](item, distance, ordering)

  /**
    * SearchResult factory.
    *
    * @param item the item
    * @param distance the distance from the search query
    * @tparam TItem type of the item returned
    * @tparam TDistance type of the distance returned by the configured distance function
    * @return a new SearchResult
    */
  def apply[TItem, TDistance](item: TItem, distance: TDistance, distanceComparator: Comparator[TDistance])
    : SearchResult[TItem, TDistance] =
      new SearchResult[TItem, TDistance](item, distance, distanceComparator)

  /**
    * Extract fields from SearchResult.
    *
    * @param result searchresult to extract the fields from
    * @tparam TItem type of the item returned
    * @tparam TDistance type of the distance returned by the configured distance function
    * @return the extracted fields
    */
  def unapply[TItem, TDistance](result: SearchResult[TItem, TDistance]): Option[(TItem, TDistance)] =
    Some(result.item -> result.distance)

}
