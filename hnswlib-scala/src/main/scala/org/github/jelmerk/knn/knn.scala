package org.github.jelmerk

import scala.collection.JavaConverters._

package object knn {

  implicit class IndexEnrichment[TId, TVector, TItem <: Item[TId, TVector], TDistance <: Comparable[TDistance]](
    index: Index[TId, TVector, TItem, TDistance]) {

    def findNearestAsSeq(vector: TVector, k: Int): Seq[SearchResult[TItem, TDistance]] =
      index.findNearest(vector, k).asScala

    def getOptionally(id: TId) = Option(index.get(id))

    def addAllAsSeq(items: Seq[TItem],
                    numThreads: Int = Runtime.getRuntime.availableProcessors,
                    listener: ProgressListener = NullProgressListener.INSTANCE,
                    progressUpdateInterval: Int = Index.DEFAULT_PROGRESS_UPDATE_INTERVAL): Unit =
      index.addAll(items.asJava, numThreads, listener, progressUpdateInterval)
  }

}
