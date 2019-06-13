package com.github.jelmerk.knn

package object scalalike {

  type Item[TId, TVector] = com.github.jelmerk.knn.Item[TId, TVector]

  type SearchResult[TItem, TDistance] = com.github.jelmerk.knn.SearchResult[TItem, TDistance]

  type FloatDistanceFunctions = com.github.jelmerk.knn.FloatDistanceFunctions

  type DoubleDistanceFunctions = com.github.jelmerk.knn.DoubleDistanceFunctions

}
