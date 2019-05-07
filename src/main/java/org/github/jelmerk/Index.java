package org.github.jelmerk;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public interface Index<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>> {

    int size();

    Collection<TItem> items();

    TItem get(TId id);

    void add(TItem item);

    default void addAll(List<TItem> items) {
        addAll(items, Runtime.getRuntime().availableProcessors());
    }

    default void addAll(List<TItem> items, int numThreads) {

        // TODO: jk maybe push it into a queue, looking up by index wont be pretty for a linked list
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);

        try {
            AtomicInteger current = new AtomicInteger(0);
            CountDownLatch latch = new CountDownLatch(numThreads);

            for (int threadId = 0; threadId < numThreads; threadId++) {

                executorService.submit(() -> {
                    while (true) {
                        int index = current.getAndIncrement();

                        if (index >= items.size()) {
                            latch.countDown();
                            break;
                        }

                        add(items.get(index));
                    }
                });
            }

            try {
                latch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
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