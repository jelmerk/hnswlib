package org.github.jelmerk;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicReference;

public interface Index<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>> {

    int size();

    Collection<TItem> items();

    TItem get(TId id);

    void add(TItem item);

    default void addAll(Collection<TItem> items) {
        addAll(items, Runtime.getRuntime().availableProcessors());
    }

    default void addAll(Collection<TItem> items, int numThreads) {

        AtomicReference<RuntimeException> throwableHolder = new AtomicReference<>();

        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);

        try {
            Queue<TItem> queue = new LinkedBlockingDeque<>(items);

            CountDownLatch latch = new CountDownLatch(numThreads);

            for (int threadId = 0; threadId < numThreads; threadId++) {

                executorService.submit(() -> {
                    TItem item;
                    while((item = queue.poll()) != null) {
                        try {
                            add(item);
                        } catch (RuntimeException t) {
                            throwableHolder.set(t);
                        }
                    }

                    latch.countDown();
                });
            }

            try {
                latch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            RuntimeException throwable = throwableHolder.get();

            if (throwable != null) {
                throw throwable;
            }

        } finally {
            executorService.shutdown();
        }
    }

    List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k);

    void save(OutputStream out) throws IOException;

    default void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }
}