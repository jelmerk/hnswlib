package com.github.jelmerk.knn.scalalike.hnsw

import java.io.{File, InputStream}
import java.nio.file.Path

import com.github.jelmerk.knn.JavaObjectSerializer
import com.github.jelmerk.knn.hnsw.{HnswIndex => JHnswIndex}
import com.github.jelmerk.knn.scalalike._

object HnswIndex {

  /**
    * Restores a [[HnswIndex]] from an InputStream.
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
    : HnswIndex[TId, TVector, TItem, TDistance] = new HnswIndex(JHnswIndex.load(inputStream, classLoader))

  /**
    * Restores a [[HnswIndex]] from a File.
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
    : HnswIndex[TId, TVector, TItem, TDistance] =
      new HnswIndex(JHnswIndex.load(file, classLoader))

  /**
    * Restores a [[HnswIndex]] from a Path.
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
    : HnswIndex[TId, TVector, TItem, TDistance] =
      new HnswIndex(JHnswIndex.load(path, classLoader))

  /**
    * Construct a new [[HnswIndex]].
    *
    * @param dimensions dimensionality of the items stored in the index
    * @param distanceFunction the distance function
    * @param maxItemCount the maximum number of elements in the index
    * @param m Sets the number of bi-directional links created for every new element during construction. Reasonable range
    *          for m is 2-100. Higher m work better on datasets with high intrinsic dimensionality and/or high recall,
    *          while low m work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter
    *          also determines the algorithm's memory consumption.
    *          As an example for d = 4 random vectors optimal m for search is somewhere around 6, while for high dimensional
    *          datasets (word embeddings, good face descriptors), higher M are required (e.g. m = 48, 64) for optimal
    *          performance at high recall. The range mM = 12-48 is ok for the most of the use cases. When m is changed one
    *          has to update the other parameters. Nonetheless, ef and efConstruction parameters can be roughly estimated by
    *          assuming that m * efConstruction is a constant.
    * @param ef The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more
    *           accurate but slower search. The value ef of can be anything between k and the size of the dataset.
    * @param efConstruction The parameter has the same meaning as ef, but controls the index time / index precision. Bigger efConstruction
    *                       leads to longer construction, but better index quality. At some point, increasing efConstruction does not
    *                       improve the quality of the index. One way to check if the selection of ef_construction was ok is to measure
    *                       a recall for M nearest neighbor search when ef = efConstruction: if the recall is lower than 0.9, then
    *                       there is room for improvement
    * @param removeEnabled    enable or disable the experimental remove operation. Indices that support removes will consume more memory
    * @param itemIdSerializer used to serialize the item key during saving of the index. when unspecified java serialization will be used.
    *                         for the fastest possible save time and smallest indices you will want to provide this
    * @param itemSerializer used to serialize the item during saving of the index. when unspecified java serialization will be used.
    *                       for the fastest possible save time and smallest indices you will want to provide this
    * @param distanceOrdering used to compare the distances returned by the distance function
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    * @return the index
    */
  def apply[TId,  TVector, TItem <: Item[TId, TVector], TDistance](
    dimensions: Int,
    distanceFunction: DistanceFunction[TVector, TDistance],
    maxItemCount: Int,
    m: Int = JHnswIndex.BuilderBase.DEFAULT_M,
    ef: Int = JHnswIndex.BuilderBase.DEFAULT_EF,
    efConstruction: Int = JHnswIndex.BuilderBase.DEFAULT_EF_CONSTRUCTION,
    removeEnabled: Boolean = JHnswIndex.BuilderBase.DEFAULT_REMOVE_ENABLED,
    itemIdSerializer: ObjectSerializer[TId] = new JavaObjectSerializer[TId],
    itemSerializer: ObjectSerializer[TItem] = new JavaObjectSerializer[TItem])(implicit distanceOrdering: Ordering[TDistance])
      : HnswIndex[TId, TVector, TItem, TDistance] = {

    val builder = JHnswIndex
      .newBuilder(dimensions, new ScalaDistanceFunctionAdapter[TVector, TDistance](distanceFunction), distanceOrdering, maxItemCount)
      .withM(m)
      .withEf(ef)
      .withEfConstruction(efConstruction)
      .withCustomSerializers(itemIdSerializer, itemSerializer)

    val jIndex =
      if(removeEnabled) builder.withRemoveEnabled().build()
      else builder.build()

    new HnswIndex[TId, TVector, TItem, TDistance](jIndex)

  }

}

/**
  * Implementation of Index that implements the hnsw algorithm.
  *
  * @see See [[https://arxiv.org/abs/1603.09320]] for more information.
  *
  * @param delegate the java index to delegate calls to
  *
  * @tparam TId Type of the external identifier of an item
  * @tparam TVector Type of the vector to perform distance calculation on
  * @tparam TItem Type of items stored in the index
  * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
  */
@SerialVersionUID(1L)
class HnswIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] private (delegate: JHnswIndex[TId, TVector, TItem, TDistance])
  extends ScalaIndexAdapter[TId, TVector, TItem ,TDistance](delegate) {

  /**
    * This distance function.
    */
  val distanceFunction: DistanceFunction[TVector, TDistance] = delegate
    .getDistanceFunction.asInstanceOf[ScalaDistanceFunctionAdapter[TVector, TDistance]].scalaFunction

  /**
    * The ordering used to compare distances
    */
  val distanceOrdering: Ordering[TDistance] = delegate.getDistanceComparator.asInstanceOf[Ordering[TDistance]]

  /**
    * The maximum number of items the index can hold.
    */
  val maxItemCount: Int = delegate.getMaxItemCount

  /**
    * True if removes are enabled for this index.
    */
  val removeEnabled: Boolean = delegate.isRemoveEnabled

  /**
    * The serializer used to serialize item id's when saving the index.
    */
  val itemIdSerializer: ObjectSerializer[TId] = delegate.getItemIdSerializer

  /**
    * The serializer used to serialize items when saving the index
    */
  val itemSerializer: ObjectSerializer[TItem] = delegate.getItemSerializer

  /**
    * The number of bi-directional links created for every new element during construction.
    */
  val m: Int = delegate.getM

  /**
    * The size of the dynamic list for the nearest neighbors (used during the search)
    */
  def ef: Int = delegate.getEf

  /**
    * Sets the size of the dynamic list for the nearest neighbors (used during the search)
    */
  def ef_= (value: Int):Unit = delegate.setEf(value)

  /**
    * Returns the parameter has the same meaning as ef, but controls the index time / index precision.
    */
  val efConstruction: Int = delegate.getEfConstruction

  /**
   * Changes the maximum capacity of the index.
   *
   * @param newSize new size of the index
   */
  def resize(newSize: Int): Unit = delegate.resize(newSize)

  /**
    * Read only view on top of this index that uses pairwise comparision when doing distance search. And as
    * such can be used as a baseline for assessing the precision of the index.
    * Searches will be really slow but give the correct result every time.
    */
  def asExactIndex: Index[TId, TVector, TItem, TDistance] = new ScalaIndexAdapter(delegate.asExactIndex())

}
