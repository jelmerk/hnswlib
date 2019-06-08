package org.github.jelmerk.knn

package object scalalike {

  type Item[TId, TVector] = org.github.jelmerk.knn.Item[TId, TVector]

  type SearchResult[TItem, TDistance] = org.github.jelmerk.knn.SearchResult[TItem, TDistance]

  type DistanceFunctions = org.github.jelmerk.knn.DistanceFunctions

}
