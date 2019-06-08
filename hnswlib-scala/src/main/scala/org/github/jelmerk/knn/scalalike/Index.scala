package org.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path

import org.github.jelmerk.knn.{Index => JIndex}

trait Index[TId, TVector, TItem <: Item[TId, TVector], TDistance] extends Serializable {

  type ProgressListener = (Int, Int) => Unit

  def add(item: TItem): Unit

  def addAll(items: Seq[TItem],
             numThreads: Int = Runtime.getRuntime.availableProcessors,
             listener: ProgressListener = (_, _) => (),
             progressUpdateInterval: Int = JIndex.DEFAULT_PROGRESS_UPDATE_INTERVAL): Unit

  def size: Int

  def apply(id: TId): TItem

  def get(id: TId): Option[TItem]

  def findNearest(vector: TVector, k: Int): Seq[SearchResult[TItem, TDistance]]

  def findNeighbours(id: TId, k: Int): Seq[SearchResult[TItem, TDistance]]

  def remove(id: TId): Boolean

  def save(out: OutputStream): Unit

  def save(out: File): Unit

  def save(path: Path): Unit

}
