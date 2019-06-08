package org.github.jelmerk.knn.scalalike.bruteforce

import java.io.{File, InputStream}
import java.nio.file.Path

import org.github.jelmerk.knn.DistanceFunction
import org.github.jelmerk.knn.bruteforce.{BruteForceIndex => JBruteForceIndex}
import org.github.jelmerk.knn.{Index => JIndex}
import org.github.jelmerk.knn.scalalike.{Index, Item}

object BruteForceIndex {

  def load[TId, TVector, TItem <: Item[TId, TVector], TDistance](inputStream: InputStream)
    : BruteForceIndex[TId, TVector, TItem, TDistance] = new BruteForceIndex(JBruteForceIndex.load(inputStream))

  def load[TId, TVector, TItem <: Item[TId, TVector], TDistance ](file: File)
    : BruteForceIndex[TId, TVector, TItem, TDistance] =
      new BruteForceIndex(JBruteForceIndex.load(file))

  def load[TId, TVector, TItem <: Item[TId, TVector], TDistance ](path: Path)
    : BruteForceIndex[TId, TVector, TItem, TDistance] =
      new BruteForceIndex(JBruteForceIndex.load(path))

  def apply[TId, TVector, TItem <: Item[TId, TVector], TDistance ]
      (distanceFunction: (TVector, TVector) => TDistance)(implicit ordering: Ordering[TDistance])
        : BruteForceIndex[TId, TVector, TItem, TDistance] = {

    val jDistanceFunction = new DistanceFunction[TVector, TDistance] {
      override def distance(u: TVector, v: TVector): TDistance = distanceFunction(u, v)
    }

    val jIndex = JBruteForceIndex.newBuilder(jDistanceFunction, ordering).build[TId, TItem]()

    new BruteForceIndex[TId, TVector, TItem, TDistance](jIndex)
  }
}

@SerialVersionUID(1L)
class BruteForceIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance ] private (
  protected val delegate: JIndex[TId, TVector, TItem, TDistance])
    extends Index[TId, TVector, TItem, TDistance]
