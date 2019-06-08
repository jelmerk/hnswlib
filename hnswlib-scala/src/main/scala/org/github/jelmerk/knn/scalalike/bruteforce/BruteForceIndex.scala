package org.github.jelmerk.knn.scalalike.bruteforce

import java.io.{File, InputStream}
import java.nio.file.Path

import org.github.jelmerk.knn.DistanceFunction
import org.github.jelmerk.knn.bruteforce.{BruteForceIndex => JBruteForceIndex}
import org.github.jelmerk.knn.scalalike.{DelegationIndex, Index, Item}

object BruteForceIndex {

  def load[TId, TVector, TItem <: Item[TId, TVector], TDistance](inputStream: InputStream)
    : Index[TId, TVector, TItem, TDistance] = new DelegationIndex(JBruteForceIndex.load(inputStream))

  def load[TId, TVector, TItem <: Item[TId, TVector], TDistance ](file: File)
    : Index[TId, TVector, TItem, TDistance] =
      new DelegationIndex(JBruteForceIndex.load(file))

  def load[TId, TVector, TItem <: Item[TId, TVector], TDistance ](path: Path)
    : Index[TId, TVector, TItem, TDistance] =
      new DelegationIndex(JBruteForceIndex.load(path))

  def apply[TId, TVector, TItem <: Item[TId, TVector], TDistance ]
      (distanceFunction: (TVector, TVector) => TDistance)(implicit ordering: Ordering[TDistance])
        : Index[TId, TVector, TItem, TDistance] = {

    val jDistanceFunction = new DistanceFunction[TVector, TDistance] {
      override def distance(u: TVector, v: TVector): TDistance = distanceFunction(u, v)
    }

    val jIndex = JBruteForceIndex.newBuilder(jDistanceFunction, ordering).build[TId, TItem]()

    new DelegationIndex[TId, TVector, TItem, TDistance](jIndex)
  }
}