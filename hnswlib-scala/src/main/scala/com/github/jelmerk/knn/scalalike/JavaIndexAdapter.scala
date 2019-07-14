package com.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path
import java.util.{ Collection => JCollection, List => JList}
import java.util.Optional
import scala.collection.JavaConverters._

import com.github.jelmerk.knn.{ProgressListener, Index => JIndex}

@SerialVersionUID(1L)
class JavaIndexAdapter[TId, TVector, TItem <: Item[TId, TVector], TDistance](delegate: Index[TId, TVector, TItem, TDistance])
  extends JIndex[TId, TVector, TItem, TDistance] {

  override def add(item: TItem): Boolean = delegate.add(item)

  override def remove(id: TId, version: Int): Boolean = delegate.remove(id, version)

  override def size(): Int = delegate.size

  override def get(id: TId): Optional[TItem] = delegate.get(id) match {
    case Some(value) => Optional.of(value);
    case _ => Optional.empty()
  }

  override def items(): JCollection[TItem] = delegate.iterator.toSeq.asJavaCollection

  override def findNearest(vector: TVector, k: Int): JList[SearchResult[TItem, TDistance]] =
    delegate.findNearest(vector, k).asJava

  override def addAll(items: JCollection[TItem]): Unit = delegate.addAll(items.asScala)

  override def addAll(items: JCollection[TItem], progressListener: ProgressListener): Unit =
    delegate.addAll(items.asScala, listener = (workDone, max) => progressListener.updateProgress(workDone, max))

  override def addAll(items: JCollection[TItem], numThreads: Int, progressListener: ProgressListener, progressUpdateInterval: Int): Unit =
    delegate.addAll(items.asScala, numThreads, (workDone, max) => progressListener.updateProgress(workDone, max), progressUpdateInterval)

  override def findNeighbors(id: TId, k: Int): JList[SearchResult[TItem, TDistance]] = delegate.findNeighbors(id, k).asJava

  override def save(out: OutputStream): Unit = delegate.save(out)

  override def save(file: File): Unit = delegate.save(file)

  override def save(path: Path): Unit = super.save(path)
}
