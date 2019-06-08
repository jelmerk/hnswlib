package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.bruteforce.BruteForceIndex;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class BruteForceIndexFastText {


    public static void main(String[] args) throws Exception {


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

        Index<String, float[], Word, Float> index = BruteForceIndex
                .newBuilder(DistanceFunctions::cosineDistance)
                    .build();

        index.addAll(words);

        index.save(new File("/Users/jkuperus/cc.nl.300.vec-bruteforce-index.ser"));
    }
}
