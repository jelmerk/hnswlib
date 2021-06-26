package com.github.jelmerk.knn;

import com.github.jelmerk.knn.util.NamedThreadFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * K-nearest neighbors search index.
 *
 * @param <TId> Type of the external identifier of an item
 * @param <TVector> Type of the vector to perform distance calculation on
 * @param <TItem> Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 *
 * @see <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbors algorithm</a>
 */
public interface Index<TId, TVector, TItem extends Item<TId, TVector>, TDistance> extends Serializable {

    /**
     * By default after indexing this many items progress will be reported to registered progress listeners.
     */
    int DEFAULT_PROGRESS_UPDATE_INTERVAL = 100_000;

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
     *
     * @return true if the item was added to the index
     * @throws IllegalArgumentException thrown when the item has the wrong dimensionality
     */
    boolean add(TItem item);

    /**
     * Removes an item from the index. If the index does not support deletes or an item with the same identifier exists
     * in the index with a higher version number, then this method will return false and the item will not be removed.
     *
     * @param id unique identifier or the item to remove
     * @param version version of the delete. If your items don't override version  use 0
     * @return {@code true} if an item was removed from the index. In case the index does not support removals this will
     *                      always be false
     */
    boolean remove(TId id, long version);

    /**
     * Check if an item is contained in this index
     *
     * @param id unique identifier of the item
     * @return true if an item is contained in this index, false otherwise
     */
    default boolean contains(TId id) {
        return get(id).isPresent();
    }

    /**
     * Add multiple items to the index
     *
     * @param items the items to add to the index
     * @throws InterruptedException thrown when the thread doing the indexing is interrupted
     */
    default void addAll(Collection<TItem> items) throws InterruptedException {
        addAll(items, NullProgressListener.INSTANCE);
    }

    /**
     * Add multiple items to the index. Reports progress to the passed in implementation of {@link ProgressListener}
     * every {@link Index#DEFAULT_PROGRESS_UPDATE_INTERVAL} elements indexed.
     *
     * @param items the items to add to the index
     * @param listener listener to report progress to
     * @throws InterruptedException thrown when the thread doing the indexing is interrupted
     */
    default void addAll(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
        addAll(items, Runtime.getRuntime().availableProcessors(), listener, DEFAULT_PROGRESS_UPDATE_INTERVAL);
    }

    /**
     * Add multiple items to the index. Reports progress to the passed in implementation of {@link ProgressListener}
     * every progressUpdateInterval elements indexed.
     *
     * @param items the items to add to the index
     * @param numThreads number of threads to use for parallel indexing
     * @param listener listener to report progress to
     * @param progressUpdateInterval after indexing this many items progress will be reported. The last element will always be reported regardless of this setting.
     * @throws InterruptedException thrown when the thread doing the indexing is interrupted
     */
    default void addAll(Collection<TItem> items, int numThreads, ProgressListener listener, int progressUpdateInterval)
            throws InterruptedException {
        ThreadPoolExecutor executorService = new ThreadPoolExecutor(numThreads, numThreads, 60L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(),
                new NamedThreadFactory("indexer-%d"));
        executorService.allowCoreThreadTimeOut(true);

        int numItems = items.size();

        AtomicInteger workDone = new AtomicInteger();

        try {
            Queue<TItem> queue = new LinkedBlockingDeque<>(items);
            List<Future<?>> futures = new ArrayList<>();

            for (int threadId = 0; threadId < numThreads; threadId++) {
                futures.add(executorService.submit(() -> {
                    TItem item;
                    while((item = queue.poll()) != null) {
                        add(item);

                        int done = workDone.incrementAndGet();
                        if (done % progressUpdateInterval == 0 || numItems == done) {
                            listener.updateProgress(done, items.size());
                        }
                    }
                }));
            }

            for(Future<?> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    throw new UncategorizedIndexException("An exception was thrown by one of the threads.", e.getCause());
                }
            }

        } finally {
            executorService.shutdown();
        }
    }

    /**
     * Returns the size of the index.
     *
     * @return size of the index
     */
    int size();

    /**
     * Returns an item by its identifier.
     *
     * @param id unique identifier or the item to return
     * @return an item
     */
    Optional<TItem> get(TId id);

    /**
     * Returns all items in the index.
     *
     * @return all items in the index
     */
    Collection<TItem> items();

    /**
     * Find the items closest to the passed in vector.
     *
     * @param vector the vector
     * @param k number of items to return
     * @return the items closest to the passed in vector
     */
    List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k);

    /**
     * Find the items closest to the item identified by the passed in id. If the id does not match an item an empty
     * list is returned. the element itself is not included in the response.
     *
     * @param id id of the item to find the neighbors of
     * @param k number of items to return
     * @return the items closest to the item
     */
    default List<SearchResult<TItem, TDistance>> findNeighbors(TId id, int k) {
        return get(id).map(item -> findNearest(item.vector(), k + 1).stream()
                .filter(result -> !result.item().id().equals(id))
                .limit(k)
                .collect(Collectors.toList()))
                .orElse(Collections.emptyList());
    }

    /**
     * Saves the index to an OutputStream. Saving is not thread safe and you should not modify the index while saving.
     *
     * @param out the output stream to write the index to
     * @throws IOException in case of I/O exception
     */
    void save(OutputStream out) throws IOException;

    /**
     * Saves the index to a file. Saving is not thread safe and you should not modify the index while saving.
     *
     * @param file file to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }

    /**
     * Saves the index to a path. Saving is not thread safe and you should not modify the index while saving.
     *
     * @param path file to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(Path path) throws IOException {
        save(Files.newOutputStream(path));
    }

}