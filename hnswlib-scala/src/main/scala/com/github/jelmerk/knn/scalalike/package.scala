package com.github.jelmerk.knn

package object scalalike {

  /**
    * Item that can be indexed
    *
    * @tparam TId The type of the vector to perform distance calculation on
    * @tparam TVector Type of the vector to perform distance calculation on
    */
  type Item[TId, TVector] = com.github.jelmerk.knn.Item[TId, TVector]

  /**
    * Result of a nearest neighbour search
    * @tparam TItem type of the item returned
    * @tparam TDistance type of the distance returned by the configured distance function
    */
  type SearchResult[TItem, TDistance] = com.github.jelmerk.knn.SearchResult[TItem, TDistance]

  /**
    * Calculates distance between 2 vectors
    */
  type DistanceFunction[TVector, TDistance] = (TVector, TVector) => TDistance

  /**
    * Called periodically to report on progress during indexing. The first argument is the number of records processed.
    * The second argument is the total number of record to index as part of this operation.
    */
  type ProgressListener = (Int, Int) => Unit
}
