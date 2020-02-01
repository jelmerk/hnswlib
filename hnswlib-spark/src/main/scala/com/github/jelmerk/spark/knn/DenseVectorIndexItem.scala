package com.github.jelmerk.spark.knn

import com.github.jelmerk.knn.scalalike.Item

/**
 * Item in an nearest neighbor search index
 *
 * @param id item identifier
 * @param vector item vector
 */
private[knn] case class DenseVectorIndexItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]]