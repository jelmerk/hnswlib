package org.github.jelmerk.knn.hnsw;

import org.github.jelmerk.knn.Index;
import org.github.jelmerk.knn.SearchResult;
import org.github.jelmerk.knn.bruteforce.BruteForceIndex;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HsnwIndexAccuracy {


    public static void main(String[] args) throws Exception {

        Random random = new Random(1000);

        int numTests = 100;
        int numResults = 10;
        int numDimensions = 300;

        float[][] tests  = new float[numTests][numDimensions];

        for (int i = 0; i < numTests; i++) {
            tests[i] = generateRandomVector(random, numDimensions);
        }

        System.out.println("Finished generating test vectors.");

        Index<String, float[], HnswIndexFastText.Word, Float> bruteForceIndex =
                BruteForceIndex.load(new File("/Users/jkuperus/cc.nl.300.vec-bruteforce-index.ser"));

        System.out.println("Finished loading brute force index");

        List<List<SearchResult<HnswIndexFastText.Word, Float>>> bruteForceResults = performQueries(bruteForceIndex, tests, numResults);

        System.out.println("Finished testing brute force index.");

        bruteForceIndex = null;

        Index<String, float[], HnswIndexFastText.Word, Float> hnswIndex =
                    HnswIndex.load(new File("/Users/jkuperus/cc.nl.300.vec-new.ser3"));

        System.out.println("Finished loading hnsw index");

        List<List<SearchResult<HnswIndexFastText.Word, Float>>> hnswResults = performQueries(hnswIndex, tests, numResults);

        System.out.println("Finished testing hnsw index.");

//        hnswIndex = null;

        System.out.println("Calculating precision.");

        double sumPrecision = 0;

        for (int i = 0; i < numTests; i++) {
            List<SearchResult<HnswIndexFastText.Word, Float>> bruteForceResult = bruteForceResults.get(i);
            List<SearchResult<HnswIndexFastText.Word, Float>> hnswResult = hnswResults.get(i);


            sumPrecision += calculatePrecision(bruteForceResult, hnswResult);
        }

        System.out.println("Precision at " + numResults + " : " + sumPrecision / (double) numTests);

    }

    private static double calculatePrecision(
            List<SearchResult<HnswIndexFastText.Word, Float>> expectedResults,
            List<SearchResult<HnswIndexFastText.Word, Float>> actualResults) {

        int correct = 0;

        for (SearchResult<HnswIndexFastText.Word, Float> expectedResult : expectedResults) {
            if (actualResults.contains(expectedResult)) {
                correct++;
            }
        }

        return (double) correct / (double) expectedResults.size();
    }

    private static List<List<SearchResult<HnswIndexFastText.Word, Float>>> performQueries(
            Index<String, float[], HnswIndexFastText.Word, Float> index,
            float[][] tests,
            int numResults) {
        List<List<SearchResult<HnswIndexFastText.Word, Float>>> results = new ArrayList<>(tests.length);

        System.out.print("Performing queries ");
        for (float[] vector : tests) {
            System.out.print(".");
            results.add(index.findNearest(vector, numResults));
        }

        System.out.print("\n");
        return results;
    }

    private static float[] generateRandomVector(Random random, int size) {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) {
            result[i] = random.nextFloat();
        }
        return result;
    }

}
