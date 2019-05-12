package org.github.jelmerk;

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

public interface Index<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>> {

    int DEFAULT_PROGRESS_UPDATE_INTERVAL = 100_000;

    int size();

    TItem get(TId id);

    void add(TItem item);

    TItem remove(TId id);

    default void addAll(Collection<TItem> items) throws InterruptedException {
        addAll(items, NullProgressListener.INSTANCE);
    }

    default void addAll(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
        addAll(items, Runtime.getRuntime().availableProcessors(), listener, DEFAULT_PROGRESS_UPDATE_INTERVAL);
    }

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

    List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k);

    default void save(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    default void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }
}