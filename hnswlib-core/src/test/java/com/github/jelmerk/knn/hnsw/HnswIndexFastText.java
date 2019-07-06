package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.FloatDistanceFunctions;
import com.github.jelmerk.knn.SearchResult;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.TimeUnit;
import static com.github.jelmerk.knn.util.VectorUtils.normalize;

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

//                words.add(new Word(word, vector));
                words.add(new Word(word, normalize(vector)));
            }

        }

        System.out.println("Loaded " + words.size() + " words.");

        int m = 16;

        HnswIndex<String, float[], Word, Float> index = HnswIndex
                .newBuilder(FloatDistanceFunctions::cosineDistance, words.size())
//                  .newBuilder(FloatDistanceFunctions::innerProduct, words.size())
//                    .withRemoveEnabled()
                    .withM(m)
                    .build();

        long start = System.currentTimeMillis();

        index.addAll(words, (workDone, max) -> System.out.printf("%d - Added %d out of %d records%n", System.currentTimeMillis(), workDone, max));

        long end = System.currentTimeMillis();

        long duration = end - start;

        System.out.println("Creating index took " + duration + " millis which is " + TimeUnit.MILLISECONDS.toMinutes(duration));

        List<SearchResult<Word, Float>> nearest = index.findNeighbors("koning", 10);

        for (SearchResult<Word, Float> result : nearest) {
            System.out.println(result.item().id() + " " + result.distance());
        }


//        index.save(new File("/Users/jkuperus/cc.nl.300.vec-new.ser3"));
    }


}
