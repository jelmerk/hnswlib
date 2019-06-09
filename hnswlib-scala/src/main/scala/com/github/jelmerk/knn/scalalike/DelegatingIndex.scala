package com.github.jelmerk.knn.scalalike

import scala.collection.JavaConverters._
import com.github.jelmerk.knn.{Index => JIndex, ProgressListener => JProgressListener}

@SerialVersionUID(1L)
class DelegatingIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance](delegate: JIndex[TId, TVector, TItem, TDistance])
  extends DelegatingReadOnlyIndex(delegate) with Index[TId, TVector, TItem, TDistance] {

  override def add(item: TItem): Unit = delegate.add(item)

  override def addAll(items: Iterable[TItem],
                      numThreads: Int = Runtime.getRuntime.availableProcessors,
                      listener: ProgressListener = (_, _) => (),
                      progressUpdateInterval: Int = JIndex.DEFAULT_PROGRESS_UPDATE_INTERVAL): Unit = {

    val progressListener: JProgressListener = new JProgressListener {
      override def updateProgress(workDone: Int, max: Int): Unit = listener.apply(workDone, max)
    }

    delegate.addAll(items.asJavaCollection, numThreads, progressListener, progressUpdateInterval)
  }

  override def remove(id: TId): Boolean = delegate.remove(id)

}