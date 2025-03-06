package com.github.jelmerk.hnswlib.scala

case class TestItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]] {
  override def dimensions: Int = vector.length
}