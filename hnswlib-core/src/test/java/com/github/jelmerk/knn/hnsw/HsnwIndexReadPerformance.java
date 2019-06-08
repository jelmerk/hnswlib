package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.SearchResult;

import java.io.File;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class HsnwIndexReadPerformance {


    public static void main(String[] args) throws Exception {


        HnswIndex<String, float[], Word, Float> index =
                    HnswIndex.load(new File("/Users/jkuperus/cc.nl.300.vec-new.ser3"));

        System.out.println("done loading index");

        List<SearchResult<Word, Float>> nearest = index.findNeighbours("koning", 10);

        for (SearchResult<Word, Float> result : nearest) {
            System.out.println(result.item().id() + " " + result.distance());
        }

//        System.exit(0);

        int numProcessors = Runtime.getRuntime().availableProcessors();

        final long numSearches = 1_000_000;
        final int numResults = 10;

        int numRandomVectors = 10_000;

        float[][] values = new float[numRandomVectors][300];

        for (int i = 0; i < numRandomVectors; i++) {
            values[i] = generateRandomVector(300);
        }

        CountDownLatch latch = new CountDownLatch(numProcessors);
        AtomicInteger counter = new AtomicInteger();

        ExecutorService executorService = Executors.newFixedThreadPool(numProcessors);
        try {
            for (int i = 0; i < numProcessors; i++) {
                executorService.submit(() -> {

                    int count;
                    while ((count = counter.getAndIncrement()) < numSearches) {
                        index.findNearest(values[count % numRandomVectors], numResults);
                    }

                    latch.countDown();
                });
            }

            long start = System.currentTimeMillis();

            latch.await();

            long end = System.currentTimeMillis();

            long duration = end - start;

            System.out.println("took " + duration + " milli seconds to do " + numSearches + " searches which is " + TimeUnit.MILLISECONDS.toSeconds(duration) + " seconds");

        } finally {
            executorService.shutdown();
        }

    }



    private static float[] generateRandomVector(int size) {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) {
            result[i] = ThreadLocalRandom.current().nextFloat();
        }
        return result;
    }

}
