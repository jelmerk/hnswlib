package org.github.jelmerk.knn;

import java.io.*;
import java.util.Collection;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * K-nearest neighbours search index.
 *
 * @param <TId> type of the external identifier of an item
 * @param <TVector> The type of the vector to perform distance calculation on
 * @param <TItem> The type of items to connect into small world.
 * @param <TDistance> The type of distance between items (expect any numeric type: float, double, decimal, int, ..).
 *
 * @see <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbors algorithm</a>
 */
public interface Index<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>> {

    /**
     * By default after indexing this many items progress will be reported to registered progress listeners.
     */
    int DEFAULT_PROGRESS_UPDATE_INTERVAL = 100_000;

    /**
     * Returns an item by its identifier.
     *
     * @param id unique identifier or the item to return
     * @return an item
     */
    TItem get(TId id);

    /**
     * Add a new item to the index
     *
     * @param item the item to add to the index
     */
    void add(TItem item);

    /**
     * Removes an item from the index.
     *
     * @param id unique identifier or the item to remove
     */
    void remove(TId id);

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
                    while((item = queue.poll()) != null) {
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
     * Find the items closest to the passed in vector.
     *
     * @param vector the vector
     * @param k number of items to return
     * @return the items closest to the passed in vector
     */
    List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k);

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
}