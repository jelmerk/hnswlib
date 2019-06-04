package org.github.jelmerk

import scala.collection.JavaConverters._

package object knn {

  implicit class IndexEnrichment[TId, TVector, TItem <: Item[TId, TVector], TDistance <: Comparable[TDistance]](
    index: Index[TId, TVector, TItem, TDistance]) {

    def findNearestAsSeq(vector: TVector, k: Int): Seq[SearchResult[TItem, TDistance]] =
      index.findNearest(vector, k).asScala

    def getOptionally(id: TId) = Option(index.get(id))

    def addAll(items: Seq[TItem]): Unit = index.addAll(items.asJava)

    def addAll(items: Seq[TItem], listener: ProgressListener): Unit = index.addAll(items.asJava, listener)

    def addAll(items: Seq[TItem], numThreads: Int, listener: ProgressListener, progressUpdateInterval: Int): Unit =
      index.addAll(items.asJava, numThreads, listener, progressUpdateInterval)
  }

}
