package com.github.jelmerk.knn.scalalike.bruteforce

import java.io.{File, InputStream}
import java.nio.file.Path

import com.github.jelmerk.knn.{Index => JIndex}
import com.github.jelmerk.knn.DistanceFunction
import com.github.jelmerk.knn.bruteforce.{BruteForceIndex => JBruteForceIndex}
import com.github.jelmerk.knn.scalalike.{ScalaIndexAdapter, Item}

object BruteForceIndex {

  /**
    * Restores a BruteForceIndex from an InputStream.
    *
    * @param inputStream InputStream to restore the index from
    * @param classLoader the classloader to use
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return The restored index
    */
  def loadFromInputStream[TId,  TVector, TItem <: Item[TId, TVector], TDistance](inputStream: InputStream,
                                                                                 classLoader: ClassLoader = Thread.currentThread.getContextClassLoader)
    : BruteForceIndex[TId, TVector, TItem, TDistance] =
      new BruteForceIndex(JBruteForceIndex.load(inputStream, classLoader))

  /**
    * Restores a BruteForceIndex from a File.
    *
    * @param file File to read from
    * @param classLoader the classloader to use
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return The restored index
    */
  def loadFromFile[TId,  TVector, TItem <: Item[TId, TVector], TDistance](file: File,
                                                                          classLoader: ClassLoader = Thread.currentThread.getContextClassLoader)
    : BruteForceIndex[TId, TVector, TItem, TDistance] =
      new BruteForceIndex(JBruteForceIndex.load(file, classLoader))

  /**
    * Restores a BruteForceIndex from a Path.
    * @param classLoader the classloader to use
    *
    * @param path Path to read from
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return The restored index
    */
  def loadFromPath[TId,  TVector, TItem <: Item[TId, TVector], TDistance](path: Path,
                                                                          classLoader: ClassLoader = Thread.currentThread.getContextClassLoader)
    : BruteForceIndex[TId, TVector, TItem, TDistance] =
      new BruteForceIndex(JBruteForceIndex.load(path, classLoader))

  /**
    * Construct a new [[BruteForceIndex]].
    *
    * @param dimensions dimensionality of the items stored in the index
    * @param distanceFunction the distance function
    * @param distanceOrdering used to compare the distances returned by the distance function
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return the index
    */
  def apply[TId, TVector, TItem <: Item[TId, TVector], TDistance ](
      dimensions: Int,
      distanceFunction: (TVector, TVector) => TDistance
    )(implicit distanceOrdering: Ordering[TDistance])
        : BruteForceIndex[TId, TVector, TItem, TDistance] = {

    val jDistanceFunction = new DistanceFunction[TVector, TDistance] {
      override def distance(u: TVector, v: TVector): TDistance = distanceFunction(u, v)
    }

    val jIndex = JBruteForceIndex.newBuilder(dimensions, jDistanceFunction, distanceOrdering).build[TId, TItem]()

    new BruteForceIndex[TId, TVector, TItem, TDistance](jIndex)
  }

  /**
   * Creates an immutable empty index.
   *
   * @tparam TId Type of the external identifier of an item
   * @tparam TVector Type of the vector to perform distance calculation on
   * @tparam TItem Type of items stored in the index
   * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
   * @return the index
   */
  def empty[TId,  TVector, TItem <: Item[TId, TVector], TDistance]: BruteForceIndex[TId, TVector, TItem, TDistance] = {
    val jIndex: JBruteForceIndex[TId, TVector, TItem, TDistance] = JBruteForceIndex.empty()
    new BruteForceIndex(jIndex)
  }
}

/**
  * Implementation of Index that uses brute force
  *
  * @param delegate the java index to delegate calls to
  *
  * @tparam TId Type of the external identifier of an item
  * @tparam TVector Type of the vector to perform distance calculation on
  * @tparam TItem Type of items stored in the index
  * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
  */
class BruteForceIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance](delegate: JIndex[TId, TVector, TItem, TDistance])
  extends ScalaIndexAdapter[TId, TVector, TItem, TDistance](delegate)