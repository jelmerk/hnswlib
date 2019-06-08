package com.github.jelmerk.knn.scalalike.hnsw

import java.io.{File, InputStream}
import java.nio.file.Path

import com.github.jelmerk.knn.hnsw.{HnswIndex => JHnswIndex}
import com.github.jelmerk.knn.DistanceFunction
import com.github.jelmerk.knn.scalalike.{DelegationIndex, Index, Item}

object HnswIndex {
  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](inputStream: InputStream)
    : Index[TId, TVector, TItem, TDistance] = new DelegationIndex(JHnswIndex.load(inputStream))

  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](file: File)
    : Index[TId, TVector, TItem, TDistance] =
      new DelegationIndex(JHnswIndex.load(file))

  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](path: Path)
    : Index[TId, TVector, TItem, TDistance] =
      new DelegationIndex(JHnswIndex.load(path))

  def apply[TId,  TVector, TItem <: Item[TId, TVector], TDistance](
    distanceFunction: (TVector, TVector) => TDistance,
    maxItemCount : Int,
    m: Int = JHnswIndex.Builder.DEFAULT_M,
    ef: Int = JHnswIndex.Builder.DEFAULT_EF,
    efConstruction: Int = JHnswIndex.Builder.DEFAULT_EF_CONSTRUCTION)(implicit ordering: Ordering[TDistance])
      : Index[TId, TVector, TItem, TDistance] = {

    val jDistanceFunction = new DistanceFunction[TVector, TDistance] {
      override def distance(u: TVector, v: TVector): TDistance = distanceFunction(u, v)
    }

    val jIndex = JHnswIndex.newBuilder(jDistanceFunction, ordering, maxItemCount)
        .withM(m)
        .withEf(ef)
        .withEfConstruction(efConstruction)
        .build[TId, TItem]()

    new DelegationIndex[TId, TVector, TItem, TDistance](jIndex)
  }

}
