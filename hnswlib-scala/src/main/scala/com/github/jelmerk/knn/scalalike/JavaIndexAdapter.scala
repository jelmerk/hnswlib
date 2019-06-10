package com.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path
import java.util.{ Collection => JCollection, List => JList}
import java.util.Optional
import scala.collection.JavaConverters._

import com.github.jelmerk.knn.{ProgressListener, Index => JIndex}

class JavaIndexAdapter[TId, TVector, TItem <: Item[TId, TVector], TDistance](delegate: Index[TId, TVector, TItem, TDistance])
  extends JIndex[TId, TVector, TItem, TDistance] {

  override def add(item: TItem): Unit = delegate.add(item)

  override def remove(id: TId): Boolean = delegate.remove(id)

  override def size(): Int = delegate.size

  override def get(id: TId): Optional[TItem] = delegate.get(id) match {
    case Some(value) => Optional.of(value);
    case _ => Optional.empty()
  }

  override def findNearest(vector: TVector, k: Int): JList[SearchResult[TItem, TDistance]] =
    delegate.findNearest(vector, k).asJava

  override def addAll(items: JCollection[TItem]): Unit = delegate.addAll(items.asScala)

  override def addAll(items: JCollection[TItem], progressListener: ProgressListener): Unit =
    delegate.addAll(items.asScala, listener = (workDone, max) => progressListener.updateProgress(workDone, max))

  override def addAll(items: JCollection[TItem], numThreads: Int, progressListener: ProgressListener, progressUpdateInterval: Int): Unit =
    delegate.addAll(items.asScala, numThreads, (workDone, max) => progressListener.updateProgress(workDone, max), progressUpdateInterval)

  override def findNeighbours(id: TId, k: Int): JList[SearchResult[TItem, TDistance]] = delegate.findNeighbours(id, k).asJava

  override def save(out: OutputStream): Unit = delegate.save(out)

  override def save(file: File): Unit = delegate.save(file)

  override def save(path: Path): Unit = super.save(path)
}
