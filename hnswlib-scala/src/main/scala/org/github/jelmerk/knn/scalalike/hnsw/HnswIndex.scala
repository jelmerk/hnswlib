package org.github.jelmerk.knn.scalalike.hnsw

import java.io.{File, InputStream}
import java.nio.file.Path

import org.github.jelmerk.knn.hnsw.{HnswIndex => JHnswIndex}
import org.github.jelmerk.knn.{DistanceFunction, Index => JIndex}
import org.github.jelmerk.knn.scalalike.{Index, Item}

object HnswIndex {
  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](inputStream: InputStream)
    : HnswIndex[TId, TVector, TItem, TDistance] = new HnswIndex(JHnswIndex.load(inputStream))

  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](file: File)
    : HnswIndex[TId, TVector, TItem, TDistance] =
      new HnswIndex(JHnswIndex.load(file))

  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](path: Path)
    : HnswIndex[TId, TVector, TItem, TDistance] =
      new HnswIndex(JHnswIndex.load(path))

  def apply[TId,  TVector, TItem <: Item[TId, TVector], TDistance](
    distanceFunction: (TVector, TVector) => TDistance,
    maxItemCount : Int,
    m: Int = JHnswIndex.Builder.DEFAULT_M,
    ef: Int = JHnswIndex.Builder.DEFAULT_EF,
    efConstruction: Int = JHnswIndex.Builder.DEFAULT_EF_CONSTRUCTION)(implicit ordering: Ordering[TDistance])
      : HnswIndex[TId, TVector, TItem, TDistance] = {

    val jDistanceFunction = new DistanceFunction[TVector, TDistance] {
      override def distance(u: TVector, v: TVector): TDistance = distanceFunction(u, v)
    }

    val jIndex = JHnswIndex.newBuilder(jDistanceFunction, ordering, maxItemCount)
        .withM(m)
        .withEf(ef)
        .withEfConstruction(efConstruction)
        .build[TId, TItem]()

    new HnswIndex[TId, TVector, TItem, TDistance](jIndex)
  }

}

@SerialVersionUID(1L)
class HnswIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] private (
  protected val delegate: JIndex[TId, TVector, TItem, TDistance])
    extends Index[TId, TVector, TItem, TDistance]
