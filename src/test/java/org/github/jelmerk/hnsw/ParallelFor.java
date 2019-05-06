package org.github.jelmerk.hnsw;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;

public class ParallelFor {

    public static void parallelFor(int start, int end, int numThreads, BiConsumer<Integer, Integer> consumer) throws Exception {



        if (numThreads == 1) {
            for (int id = start; id < end; id++) {
                consumer.accept(id, 0);

            }
        } else {

            // TODO i dont think we want to initialize a new one every time
            ExecutorService executorService = Executors.newFixedThreadPool(numThreads);

            try {

                AtomicInteger current = new AtomicInteger(start);

                CountDownLatch latch = new CountDownLatch(numThreads);

                AtomicReference<Exception> exceptionHolder = new AtomicReference<>();

                for (int threadId = 0; threadId < numThreads; threadId++) {

                    final int finalThreadId = threadId;

                    executorService.submit(() -> {
                        while (true) {
                            int id = current.incrementAndGet();

                            if (id >= end) {
                                latch.countDown();
                                break;
                            }

                            try {
                                consumer.accept(id, finalThreadId);
                            } catch (Exception e) {
                                exceptionHolder.set(e);
                                current.set(end);
                            }
                        }
                    });

                }

                try {
                    latch.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }

                Exception exception = exceptionHolder.get();

                if (exception != null) {
                    throw exception;
                }
            } finally {
                executorService.shutdown();
            }

        }
    }


}
