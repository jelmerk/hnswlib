package com.github.jelmerk.knn.scalalike

import com.github.jelmerk.knn.{Index => JIndex}

/**
  * K-nearest neighbours search index.
  *
  * @tparam TId type of the external identifier of an item
  * @tparam TVector The type of the vector to perform distance calculation on
  * @tparam TItem The type of items to connect into small world.
  * @tparam TDistance The type of distance between items (expect any numeric type: float, double, int, ..).
  *
  * @see See [[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]] for more information.
  */
trait Index[TId, TVector, TItem <: Item[TId, TVector], TDistance]
  extends ReadOnlyIndex[TId, TVector, TItem, TDistance] {

  /**
    * Called periodically to report on progress during indexing. The first argument is the number of records processed.
    * The second argument is the total number of record to index as part of this operation.
    */
  type ProgressListener = (Int, Int) => Unit

  /**
    * Add a new item to the index.
    *
    * @param item the item to add to the index
    */
  def add(item: TItem): Unit

  /**
    * Add multiple items to the index. Reports progress to the passed in progress listener
    * every progressUpdateInterval elements indexed.

    * @param items the items to add to the index
    * @param numThreads number of threads to use for parallel indexing
    * @param listener listener to report progress to
    * @param progressUpdateInterval after indexing this many items progress will be reported
    */
  def addAll(items: Iterable[TItem],
             numThreads: Int = Runtime.getRuntime.availableProcessors,
             listener: ProgressListener = (_, _) => (),
             progressUpdateInterval: Int = JIndex.DEFAULT_PROGRESS_UPDATE_INTERVAL): Unit

  /**
    * Removes an item from the index.
    * @param id unique identifier or the item to remove
    * @return true if this list contained the specified element
    */
  def remove(id: TId): Boolean


}
