package com.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path
import java.util.{ Collection => JCollection, List => JList}
import java.util.Optional
import scala.collection.JavaConverters._

import com.github.jelmerk.knn.{ProgressListener => JProgressListener, Index => JIndex}

/**
  * Adapts the interface of a java Index to that of the scala index.
  *
  * @param delegate scala class this adapter class delegates to
  * @tparam TId Type of the external identifier of an item
  * @tparam TVector Type of the vector to perform distance calculation on
  * @tparam TItem Type of items stored in the index
  * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
  */
@SerialVersionUID(1L)
private[scalalike] class JavaIndexAdapter[TId, TVector, TItem <: Item[TId, TVector], TDistance](val delegate: Index[TId, TVector, TItem, TDistance])
  extends JIndex[TId, TVector, TItem, TDistance] {

  override def add(item: TItem): Boolean = delegate.add(item)

  override def remove(id: TId, version: Long): Boolean = delegate.remove(id, version)

  override def size(): Int = delegate.size

  override def get(id: TId): Optional[TItem] = delegate.get(id) match {
    case Some(value) => Optional.of(value);
    case _ => Optional.empty()
  }

  override def contains(id: TId): Boolean = delegate.contains(id)

  override def items(): JCollection[TItem] = delegate.iterator.toSeq.asJavaCollection

  override def findNearest(vector: TVector, k: Int): JList[SearchResult[TItem, TDistance]] =
    delegate.findNearest(vector, k).asJava

  override def addAll(items: JCollection[TItem]): Unit = delegate.addAll(items.asScala)

  override def addAll(items: JCollection[TItem], progressListener: JProgressListener): Unit =
    delegate.addAll(items.asScala, listener = new JavaProgressListenerAdapter(progressListener))

  override def addAll(items: JCollection[TItem], numThreads: Int, progressListener: JProgressListener, progressUpdateInterval: Int): Unit =
    delegate.addAll(items.asScala, numThreads, new JavaProgressListenerAdapter(progressListener), progressUpdateInterval)

  override def findNeighbors(id: TId, k: Int): JList[SearchResult[TItem, TDistance]] = delegate.findNeighbors(id, k).asJava

  override def save(out: OutputStream): Unit = delegate.save(out)

  override def save(file: File): Unit = delegate.save(file)

  override def save(path: Path): Unit = super.save(path)
}

