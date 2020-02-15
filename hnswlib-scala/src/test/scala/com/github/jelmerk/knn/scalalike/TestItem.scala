package com.github.jelmerk.knn.scalalike

case class TestItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]] {
  override def dimensions: Int = vector.length
}