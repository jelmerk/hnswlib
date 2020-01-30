package com.github.jelmerk.knn.scalalike

import java.util.{List => JList}

import scala.collection.immutable
import scala.collection.JavaConverters._

/**
  * Treats a java list as an immutable sequence. This is unsafe because the underlying collection could still be
  * mutated so use it judiciously.
  *
  * @param delegate the java list
  * @tparam A type of the elements in this sequence
  */
private[scalalike] class UnsafeImmutableJListWrapper[A](delegate: JList[A]) extends immutable.Seq[A] {

  override def length: Int = delegate.size()

  override def apply(idx: Int): A = delegate.get(idx)

  override def iterator: Iterator[A] = delegate.iterator.asScala
}
