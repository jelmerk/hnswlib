package com.github.jelmerk.knn.scalalike

import java.io.{File, OutputStream}
import java.nio.file.Path

import scala.collection.immutable.Seq

import com.github.jelmerk.knn.{Index => JIndex}

/**
  * K-nearest neighbors search index.
  *
  * @tparam TId Type of the external identifier of an item
  * @tparam TVector Type of the vector to perform distance calculation on
  * @tparam TItem Type of items stored in the index
  * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..).
  *
  * @see See [[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]] for more information.
  */
trait Index[TId, TVector, TItem <: Item[TId, TVector], TDistance] extends Iterable[TItem] with Serializable {

  /**
    * By default after indexing this many items progress will be reported to registered progress listeners.
    */
  val DefaultProgressUpdateInterval: Int = JIndex.DEFAULT_PROGRESS_UPDATE_INTERVAL

  /**
    * Add a new item to the index. If an item with the same identifier already exists in the index then :
    *
    * If deletes are disabled on this index the method will return false and the item will not be updated.
    *
    * If deletes are enabled and the version of the item has is higher version than that of the item currently stored
    * in the index the old item will be removed and the new item added, otherwise this method will return false and the
    * item will not be updated.
    *
    * @param item the item to add to the index
    * @return true if the item was added to the index
    * @throws IllegalArgumentException thrown when the item has the wrong dimensionality
    */
  def add(item: TItem): Boolean

  /**
    * Add multiple items to the index. Reports progress to the passed in progress listener
    * every progressUpdateInterval elements indexed.

    * @param items the items to add to the index
    * @param numThreads number of threads to use for parallel indexing
    * @param listener listener to report progress to
    * @param progressUpdateInterval after indexing this many items progress will be reported
    */
  def addAll(items: Iterable[TItem],
             numThreads: Int = sys.runtime.availableProcessors,
             listener: ProgressListener = (_, _) => (),
             progressUpdateInterval: Int = DefaultProgressUpdateInterval): Unit

  /**
    * Removes an item from the index. If the index does not support deletes or an item with the same identifier exists
    * in the index with a higher version number, then this method will return false and the item will not be removed.
    *
    * @param id unique identifier or the item to remove
    * @param version version of the delete. If your items don't override version use 0
    * @return true if an item was removed from the index
    */
  def remove(id: TId, version: Long): Boolean

  /**
    * Returns the size of the index.
    *
    * @return size of the index
    */
  def size: Int

  /**
    * Returns an item by its identifier. If the item does not exist in the index a NoSuchElementException is thrown.
    *
    * @param id unique identifier of the item to return
    * @return the item
    */
  def apply(id: TId): TItem

  /**
    * Optionally return an item by its identifier
    * @param id unique identifier of the item to return
    *
    * @return the item
    */
  def get(id: TId): Option[TItem]

  /**
    * Check if an item is contained in this index
    *
    * @param id unique identifier of the item
    * @return true if an item is contained in this index, false otherwise
    */
  def contains(id: TId): Boolean

  /**
    * Find the items closest to the passed in vector.
    *
    * @param vector the vector
    * @param k number of items to return
    * @return the items closest to the passed in vector
    */
  def findNearest(vector: TVector, k: Int): Seq[SearchResult[TItem, TDistance]]

  /**
    * Find the items closest to the item identified by the passed in id. If the id does not match an item an empty
    * list is returned. the element itself is not included in the response.

    * @param id of the item to find the neighbors of
    * @param k number of neighbors to return
    * @return the items closest to the item
    */
  def findNeighbors(id: TId, k: Int): Seq[SearchResult[TItem, TDistance]]

  /**
    * Saves the index to an OutputStream. Saving is not thread safe and you should not modify the index while saving.
    *
    * @param out the output stream to write the index to
    */
  def save(out: OutputStream): Unit

  /**
    * Saves the index to a file. Saving is not thread safe and you should not modify the index while saving.
    *
    * @param file file to write the index to
    */
  def save(file: File): Unit

  /**
    * Saves the index to a path. Saving is not thread safe and you should not modify the index while saving.
    *
    * @param path path to write the index to
    */
  def save(path: Path): Unit

}
