package org.github.jelmerk.hnsw;

import org.github.jelmerk.SearchResult;

import java.io.File;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

public class HsnwIndexReadPerformance {


    public static void main(String[] args) throws Exception {


        HnswIndex<String, float[], HnswIndexFastText.Word, Float> index =
//                HnswIndex.load(new File("/Users/jkuperus/cc.nl.300.vec.ser"));
                    HnswIndex.load(new File("/Users/jkuperus/cc.nl.300.vec-new.ser3"));

        System.out.println("done loading index");

        HnswIndexFastText.Word word = index.get("koning");

//        List<SearchResult<HnswIndexFastText.Word, Float>> nearest = index.findNearest(word.getVector(), 10);
        List<SearchResult<HnswIndexFastText.Word, Float>> nearest = index.findNearest2(word.getVector(), 50);

        for (SearchResult<HnswIndexFastText.Word, Float> result : nearest) {
            System.out.println(result.getItem().getId() + " " + result.getDistance());
        }

        System.exit(0);

        int numProcessors = Runtime.getRuntime().availableProcessors();

        final long numSearches = 1_000_000;
        final int numResults = 10;

        int numRandomVectors = 10_000;

        float[][] values = new float[numRandomVectors][300];

        for (int i = 0; i < numRandomVectors; i++) {
            values[i] = generateRandomVector(300);
        }

        CosineDistance.counter.set(0);

        CountDownLatch latch = new CountDownLatch(numProcessors);
        AtomicInteger counter = new AtomicInteger();

        ExecutorService executorService = Executors.newFixedThreadPool(numProcessors);
        try {
            for (int i = 0; i < numProcessors; i++) {
                executorService.submit(() -> {

                    int count;
                    while ((count = counter.getAndIncrement()) < numSearches) {
                        index.findNearest2(values[count % numRandomVectors], numResults);
//                        index.findNearest(values[count % numRandomVectors], numResults);
                    }

                    latch.countDown();
                });
            }

            long start = System.currentTimeMillis();

            latch.await();

            long end = System.currentTimeMillis();

            long duration = end - start;

            System.out.println("took " + duration + " milli seconds");

            System.out.println("cosine distances : " + CosineDistance.counter.get());
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
