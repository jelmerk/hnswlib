package com.github.jelmerk.knn.scalalike.bruteforce

import java.io.{File, InputStream}
import java.nio.file.Path

import com.github.jelmerk.knn.DistanceFunction
import com.github.jelmerk.knn.bruteforce.{BruteForceIndex => JBruteForceIndex}
import com.github.jelmerk.knn.scalalike.{DelegatingIndex, Index, Item}

object BruteForceIndex {

  /**
    * Restores a BruteForceIndex from an InputStream.
    *
    * @param inputStream InputStream to restore the index from
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return The restored index
    */
  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](inputStream: InputStream)
    : Index[TId, TVector, TItem, TDistance] =
      new DelegatingIndex(JBruteForceIndex.load(inputStream))

  /**
    * Restores a BruteForceIndex from a File.
    *
    * @param file File to read from
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return The restored index
    */
  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](file: File)
    : Index[TId, TVector, TItem, TDistance] =
      new DelegatingIndex(JBruteForceIndex.load(file))

  /**
    * Restores a BruteForceIndex from a Path.
    *
    * @param path Path to read from
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return The restored index
    */
  def load[TId,  TVector, TItem <: Item[TId, TVector], TDistance](path: Path)
    : Index[TId, TVector, TItem, TDistance] =
      new DelegatingIndex(JBruteForceIndex.load(path))

  def apply[TId, TVector, TItem <: Item[TId, TVector], TDistance ]
      (distanceFunction: (TVector, TVector) => TDistance)(implicit ordering: Ordering[TDistance])
        : Index[TId, TVector, TItem, TDistance] = {

    val jDistanceFunction = new DistanceFunction[TVector, TDistance] {
      override def distance(u: TVector, v: TVector): TDistance = distanceFunction(u, v)
    }

    val jIndex = JBruteForceIndex.newBuilder(jDistanceFunction, ordering).build[TId, TItem]()

    new DelegatingIndex[TId, TVector, TItem, TDistance](jIndex)
  }
}