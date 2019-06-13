package com.github.jelmerk.knn;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

/**
 * Read write K-nearest neighbors search index.
 *
 * @param <TId> Type of the external identifier of an item
 * @param <TVector> Type of the vector to perform distance calculation on
 * @param <TItem> Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 *
 * @see <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbors algorithm</a>
 */
public interface Index<TId, TVector, TItem extends Item<TId, TVector>, TDistance> {

    /**
     * By default after indexing this many items progress will be reported to registered progress listeners.
     */
    int DEFAULT_PROGRESS_UPDATE_INTERVAL = 100_000;

    /**
     * Add a new item to the index. If the item already exists in the index the old item will first be removed from the
     * index. for this removes need to be enabled for the index.
     *
     * @param item the item to add to the index
     */
    void add(TItem item);

    /**
     * Removes an item from the index.
     *
     * @param id unique identifier or the item to remove
     * @return {@code true} if this list contained the specified element
     */
    boolean remove(TId id);

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
     * @param progressUpdateInterval after indexing this many items progress will be reported
     * @throws InterruptedException thrown when the thread doing the indexing is interrupted
     */
    default void addAll(Collection<TItem> items, int numThreads, ProgressListener listener, int progressUpdateInterval)
            throws InterruptedException {

        AtomicReference<RuntimeException> throwableHolder = new AtomicReference<>();

        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);

        AtomicInteger workDone = new AtomicInteger();

        try {
            Queue<TItem> queue = new LinkedBlockingDeque<>(items);

            CountDownLatch latch = new CountDownLatch(numThreads);

            for (int threadId = 0; threadId < numThreads; threadId++) {

                executorService.submit(() -> {
                    TItem item;
                    while(throwableHolder.get() == null && (item = queue.poll()) != null) {
                        try {
                            add(item);

                            int done = workDone.incrementAndGet();

                            if (done % progressUpdateInterval == 0) {
                                listener.updateProgress(done, items.size());
                            }

                        } catch (RuntimeException t) {
                            throwableHolder.set(t);
                        }
                    }

                    latch.countDown();
                });
            }

            latch.await();

            RuntimeException throwable = throwableHolder.get();

            if (throwable != null) {
                throw throwable;
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
     * Saves the index to an OutputStream.
     *
     * @param out the output stream to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    /**
     * Saves the index to a file.
     *
     * @param file file to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }

    /**
     * Saves the index to a path.
     *
     * @param path file to write the index to
     * @throws IOException in case of I/O exception
     */
    default void save(Path path) throws IOException {
        save(Files.newOutputStream(path));
    }

}