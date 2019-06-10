package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.SearchResult;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class HsnwIndexAccuracy {


    public static void main(String[] args) throws Exception {

        int numResults = 10;

        List<String> words = Arrays.asList(
            "fiets", "appel", "trein", "computer", "schoen", "trui", "rood", "lopen", "werken", "eten",
            "kast", "keuken", "huis", "hamer", "tas", "kat", "koe", "data", "koning"
        );

        HnswIndex<String, float[], Word, Float> hnswIndex =
                HnswIndex.load(new File("/Users/jkuperus/cc.nl.300.vec-new.ser3"));

        System.out.println("Finished loading hnsw index");

        Index<String, float[], Word, Float> bruteForceIndex = hnswIndex.asExactIndex();

        System.out.println("Done picking some random words from the index to use as entrypoints.");

        List<List<SearchResult<Word, Float>>> bruteForceResults = performQueries(bruteForceIndex, words, numResults);

        System.out.println("Finished testing brute force index.");


        List<List<SearchResult<Word, Float>>> hnswResults = performQueries(hnswIndex, words, numResults);

        System.out.println("Finished testing hnsw index.");

        System.out.println("Calculating precision.");

        double sumPrecision = 0;

        for (int i = 0; i < words.size(); i++) {
            List<SearchResult<Word, Float>> bruteForceResult = bruteForceResults.get(i);
            List<SearchResult<Word, Float>> hnswResult = hnswResults.get(i);

            sumPrecision += calculatePrecision(bruteForceResult, hnswResult);
        }

        System.out.println("Precision at " + numResults + " : " + sumPrecision / (double) words.size());

    }

    private static double calculatePrecision(
            List<SearchResult<Word, Float>> expectedResults,
            List<SearchResult<Word, Float>> actualResults) {

        int correct = expectedResults.stream().mapToInt(r -> actualResults.contains(r) ? 1 : 0).sum();
        return (double) correct / (double) expectedResults.size();
    }

    private static List<List<SearchResult<Word, Float>>> performQueries(
            Index<String, float[], Word, Float> index,
            List<String> words,
            int numResults) {
        List<List<SearchResult<Word, Float>>> results = new ArrayList<>(words.size());

        System.out.print("Performing queries ");
        for (String id : words) {
            System.out.print(".");
            results.add(index.findNeighbours(id, numResults));
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
