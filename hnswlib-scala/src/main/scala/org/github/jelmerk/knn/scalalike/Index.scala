package org.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path

import scala.collection.JavaConverters._
import org.github.jelmerk.knn.{Index => JIndex, ProgressListener => JProgressListener}

trait Index[TId, TVector, TItem <: Item[TId, TVector], TDistance] extends Serializable {

  type ProgressListener = (Int, Int) => Unit

  protected def delegate: JIndex[TId, TVector, TItem, TDistance]

  def add(item: TItem): Unit = delegate.add(item)

  def addAll(items: Seq[TItem],
             numThreads: Int = Runtime.getRuntime.availableProcessors,
             listener: ProgressListener = (_, _) => (),
             progressUpdateInterval: Int = JIndex.DEFAULT_PROGRESS_UPDATE_INTERVAL): Unit = {

    val progressListener: JProgressListener = new JProgressListener {
      override def updateProgress(workDone: Int, max: Int): Unit = listener.apply(workDone, max)
    }

    delegate.addAll(items.asJava, numThreads, progressListener, progressUpdateInterval)
  }

  def size: Int = delegate.size

  def apply(id: TId): TItem = get(id).getOrElse(throw new NoSuchElementException)

  def get(id: TId): Option[TItem] = Option(delegate.get(id))

  def findNearest(vector: TVector, k: Int): Seq[SearchResult[TItem, TDistance]] =
    delegate.findNearest(vector, k).asScala

  def remove(id: TId): Boolean = delegate.remove(id)

  def save(out: OutputStream): Unit = delegate.save(out)

  def save(out: File): Unit = delegate.save(out)

  def save(path: Path): Unit = delegate.save(path)

}
