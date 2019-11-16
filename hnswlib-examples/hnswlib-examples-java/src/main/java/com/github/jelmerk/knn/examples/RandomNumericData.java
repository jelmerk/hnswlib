package com.github.jelmerk.knn.examples;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Example application that generates random sample data, inserts them into an hnsw index and makes
 * nearest neighbor query for any random number picked from graph
 * @link https://github.com/nmslib/hnswlib/blob/master/README.md
 * Note: This is not standalone example it requires PointSet class to run.
 */
public class RandomNumericData {
    private static Point random_create_TC(Random rng) {
        int id = rng.nextInt() & Integer.MAX_VALUE;  // Only positive integer ID's of random points
        double x = rng.nextDouble();
        double y = rng.nextDouble();

        return new Point(id, new double[]{x, y});
    }

    public static void main(String[] args) throws Exception {
        int seed = (int) (Math.random() * Integer.MAX_VALUE);
        Random rng = new Random(seed);  // rng = Random Number Generator
        ArrayList<Point> random_points = new ArrayList<Point>();
        int size = 10;  // Size of Random Points Dataset
        for (int i = 0; i < size; i++) {
            Point point = random_create_TC(rng);
            random_points.add(point);
        }
        System.out.println("Constructing index of Random Points. . .");

        HnswIndex<Integer, double[], Point, Double> hnswIndex = HnswIndex
                .newBuilder(DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE, random_points.size())
                .withM(16)
                .withEf(200)
                .withEfConstruction(200)
                .build();

        hnswIndex.addAll(random_points, (workDone, max) -> System.out.printf("Added %d out of %d test cases to the index.%n", workDone, max));
        System.out.printf("Creating index with %d test cases\n", hnswIndex.size());
        Index<Integer, double[], Point, Double> groundTruthIndex = hnswIndex.asExactIndex();
        System.out.println("Index created!\n");

        int k = 1;  // No. of nearest neighbors to find
        // Get any random point from Dataset to server as query point
        int index = rng.nextInt(random_points.size());
        Point query_point = random_points.get(index);


        List<SearchResult<Point, Double>> approximateResults = hnswIndex.findNeighbors(query_point.id(), k);
        List<SearchResult<Point, Double>> groundTruthResults = groundTruthIndex.findNeighbors(query_point.id(), k);

        System.out.println("Query Point:");
        System.out.println(query_point.printPoint());

        System.out.println("\nApproximate search Result: ID, Nearest Point, Distance");
        for (SearchResult<Point, Double> result : approximateResults) {
            System.out.printf("%s, %s, %.4f\n", result.item().id(), result.item().printPoint(), result.distance());
        }

        System.out.println("\nExact search Result: ID, Nearest Point, Distance");
        for (SearchResult<Point, Double> result : groundTruthResults) {
            System.out.printf("%s, %s, %.4f\n", result.item().id(), result.item().printPoint(), result.distance());
        }

        int correct = groundTruthResults.stream().mapToInt(r -> approximateResults.contains(r) ? 1 : 0).sum();
        System.out.printf("%nAccuracy : %.4f%n%n", correct / (double) groundTruthResults.size());


        System.out.println("Printing All points...");
        for (Point pt : random_points) {
            System.out.println("("+pt.vector()[0]+","+pt.vector()[1]+")");;
        }

        /*
        *You can copy these printed points and query set to graphing app i.e (www.desmos.com/calculator)
        * to visualize random point and nearest neighbor
         */
        System.out.println("end of program");
    }
}
