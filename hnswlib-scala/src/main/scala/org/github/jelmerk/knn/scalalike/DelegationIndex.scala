package org.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path

import scala.collection.JavaConverters._
import org.github.jelmerk.knn.{Index => JIndex, ProgressListener => JProgressListener}

@SerialVersionUID(1L)
class DelegationIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance](delegate: JIndex[TId, TVector, TItem, TDistance])
  extends Index[TId, TVector, TItem, TDistance] {

  override def add(item: TItem): Unit = delegate.add(item)

  override def addAll(items: Seq[TItem],
                      numThreads: Int = Runtime.getRuntime.availableProcessors,
                      listener: ProgressListener = (_, _) => (),
                      progressUpdateInterval: Int = JIndex.DEFAULT_PROGRESS_UPDATE_INTERVAL): Unit = {

    val progressListener: JProgressListener = new JProgressListener {
      override def updateProgress(workDone: Int, max: Int): Unit = listener.apply(workDone, max)
    }

    delegate.addAll(items.asJava, numThreads, progressListener, progressUpdateInterval)
  }

  override def size: Int = delegate.size

  override def apply(id: TId): TItem = get(id).getOrElse(throw new NoSuchElementException)

  override def get(id: TId): Option[TItem] = Option(delegate.get(id).orElse(null.asInstanceOf[TItem]))

  override def findNearest(vector: TVector, k: Int): Seq[SearchResult[TItem, TDistance]] =
    delegate.findNearest(vector, k).asScala

  override def findNeighbours(id: TId, k: Int): Seq[SearchResult[TItem, TDistance]] =
    delegate.findNeighbours(id, k).asScala

  override def remove(id: TId): Boolean = delegate.remove(id)

  override def save(out: OutputStream): Unit = delegate.save(out)

  override def save(out: File): Unit = delegate.save(out)

  override def save(path: Path): Unit = delegate.save(path)

}