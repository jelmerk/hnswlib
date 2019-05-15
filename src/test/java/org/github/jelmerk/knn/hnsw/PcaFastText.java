package org.github.jelmerk.knn.hnsw;

import smile.projection.PCA;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PcaFastText {


    public static void main(String[] args) throws Exception {

        int numItems = 2_000_000;
        int dimensions = 300;

        String[] words = new String[numItems];
        double[][] points = new double[numItems][dimensions];

        int row = 0;
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

                double[] vector = new double[tokens.length - 1];

                for (int i = 1; i < tokens.length - 1; i++) {
                    vector[i] = Double.valueOf(tokens[i]);
                }

                words[row] = word;
                points[row] = vector;

                row++;
            }

        }

        PCA pca = new PCA(points);
        pca.setProjection(90);

        double[][] newData = pca.getProjection().array();

        for (int i = 0; i < 100; i++) {

            System.out.println(Arrays.toString(newData[i]));

        }



    }
}
