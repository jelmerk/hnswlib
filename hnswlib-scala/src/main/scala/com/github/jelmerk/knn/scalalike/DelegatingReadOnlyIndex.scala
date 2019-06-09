package com.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path

import com.github.jelmerk.knn.{ReadOnlyIndex => JReadOnlyIndex}

import scala.collection.JavaConverters._

@SerialVersionUID(1L)
class DelegatingReadOnlyIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance](delegate: JReadOnlyIndex[TId, TVector, TItem, TDistance])
  extends ReadOnlyIndex[TId, TVector, TItem, TDistance] {

  override def size: Int = delegate.size

  override def apply(id: TId): TItem = get(id).getOrElse(throw new NoSuchElementException)

  override def get(id: TId): Option[TItem] = Option(delegate.get(id).orElse(null.asInstanceOf[TItem]))

  override def findNearest(vector: TVector, k: Int): Seq[SearchResult[TItem, TDistance]] =
    delegate.findNearest(vector, k).asScala

  override def findNeighbours(id: TId, k: Int): Seq[SearchResult[TItem, TDistance]] =
    delegate.findNeighbours(id, k).asScala

  override def save(out: OutputStream): Unit = delegate.save(out)

  override def save(file: File): Unit = delegate.save(file)

  override def save(path: Path): Unit = delegate.save(path)

}