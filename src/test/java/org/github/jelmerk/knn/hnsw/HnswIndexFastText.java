package org.github.jelmerk.knn.hnsw;

import org.github.jelmerk.knn.DistanceFunctions;
import org.github.jelmerk.knn.Item;
import org.github.jelmerk.knn.SearchResult;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class HnswIndexFastText {


    public static void main(String[] args) throws Exception{

        List<Word> words = new ArrayList<>();

        boolean first = true;
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("/Users/jkuperus/Downloads/cc.nl.300.vec")))) {
            String line;

            while((line = reader.readLine()) != null) {

                if (first) {
                    first = false;
                    continue;
                }

                String[] tokens = line.split(" ");

                String word = tokens[0];

                float[] vector = new float[tokens.length - 1];

                for (int i = 1; i < tokens.length - 1; i++) {
                    vector[i] = Float.valueOf(tokens[i]);
                }

                words.add(new Word(word, vector));
            }

        }

        System.out.println("Loaded " + words.size() + " words.");




        int m = 16;

        HnswIndex<String, float[], Word, Float> index =
                new HnswIndex.Builder<>(DistanceFunctions::cosineDistance, words.size())
                        .setM(m)
                        .build();

//        Index<String, float[], Word, Float> index =
//                new BruteForceIndex.Builder<>(CosineDistance::cosineDistance)
//                        .build();


        long start = System.currentTimeMillis();

        index.addAll(words, (workDone, max) -> System.out.printf("%d - Added %d out of %d records%n", System.currentTimeMillis(), workDone, max));

        long end = System.currentTimeMillis();

        long duration = end - start;

        System.out.println("Creating index took " + duration + " millis which is " + TimeUnit.MILLISECONDS.toMinutes(duration));

//        Index<String, float[], Word, Float> index = HnswIndex.load(new File("/Users/jkuperus/cc.nl.300.vec.ser"));


        Word item = index.get("koning");

        List<SearchResult<Word, Float>> nearest = index.findNearest(item.vector, 10);

        for (SearchResult<Word, Float> result : nearest) {
            System.out.println(result.getItem().getId() + " " + result.getDistance());
        }

//        for (SearchResult<Word, Float> result : nearest) {
//
//            System.out.println(result.getItem().getId() + " " + result.getDistance());
//        }

        index.save(new File("/Users/jkuperus/cc.nl.300.vec-new.ser3"));
    }

    static class Word implements Item<String, float[]> {

        private String id;

        private float[] vector;


        public Word(String id, float[] vector) {
            this.id = id;
            this.vector = vector;
        }

        @Override
        public String getId() {
            return id;
        }

        @Override
        public float[] getVector() {
            return vector;
        }

        @Override
        public String toString() {
            return "Word{" +
                    "id='" + id + '\'' +
                    ", vector=" + Arrays.toString(vector) +
                    '}';
        }
    }

}
