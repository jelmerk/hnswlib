package com.github.jelmerk.knn;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

public class Jdk17DistanceFunctionsTest {

    private final double error = 1e-3;

    private final float[] floatVector1 = createRandomVector(66);
    private final float[] floatVector2 = createRandomVector(66);

    @Test
    void float128CosineDistance() {
        float result1 = DistanceFunctions.FLOAT_COSINE_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_COSINE_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float256CosineDistance() {
        float result1 = DistanceFunctions.FLOAT_COSINE_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_COSINE_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float128InnerProduct() {
        float result1 = DistanceFunctions.FLOAT_INNER_PRODUCT.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_INNER_PRODUCT.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float256InnerProduct() {
        float result1 = DistanceFunctions.FLOAT_INNER_PRODUCT.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_INNER_PRODUCT.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float128EuclideanDistance() {
        float result1 = DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float256EuclideanDistance() {
        float result1 = DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float128ManhattanDistance() {
        float result1 = DistanceFunctions.FLOAT_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float256ManhattanDistance() {
        float result1 = DistanceFunctions.FLOAT_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float128CanberraDistance() {
        float result1 = DistanceFunctions.FLOAT_CANBERRA_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_CANBERRA_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2, closeTo(result1, error));
    }

    @Test
    void float256CanberraDistance() {
        float result1 = DistanceFunctions.FLOAT_CANBERRA_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_CANBERRA_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2, closeTo(result1, error));
    }

    @Test
    void float128BrayCurtisDistance() {
        float result1 = DistanceFunctions.FLOAT_BRAY_CURTIS_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_BRAY_CURTIS_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2, closeTo(result1, error));
    }

    @Test
    void float256BrayCurtisDistance() {
        float result1 = DistanceFunctions.FLOAT_BRAY_CURTIS_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_BRAY_CURTIS_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2, closeTo(result1, error));
    }

    @Test
    @Disabled
    void performance() {

        int numCompares = 100_000_000;

        List<Pair<DistanceFunction<float[], Float>, DistanceFunction<float[], Float>>> pairs = new ArrayList<>();

        pairs.add(new Pair<>("inner product float 128", DistanceFunctions.FLOAT_INNER_PRODUCT, Jdk17DistanceFunctions.VECTOR_FLOAT_128_INNER_PRODUCT));
        pairs.add(new Pair<>("inner product float 256", DistanceFunctions.FLOAT_INNER_PRODUCT, Jdk17DistanceFunctions.VECTOR_FLOAT_256_INNER_PRODUCT));

        pairs.add(new Pair<>("cosine float 128", DistanceFunctions.FLOAT_COSINE_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_128_COSINE_DISTANCE));
        pairs.add(new Pair<>("cosine float 256", DistanceFunctions.FLOAT_COSINE_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_256_COSINE_DISTANCE));

        pairs.add(new Pair<>("euclidean float 128", DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_128_EUCLIDEAN_DISTANCE));
        pairs.add(new Pair<>("euclidean float 256", DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_256_EUCLIDEAN_DISTANCE));

        pairs.add(new Pair<>("manhattan float 128", DistanceFunctions.FLOAT_MANHATTAN_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_128_MANHATTAN_DISTANCE));
        pairs.add(new Pair<>("manhattan float 256", DistanceFunctions.FLOAT_MANHATTAN_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_256_MANHATTAN_DISTANCE));

        pairs.add(new Pair<>("bray curtis float 128", DistanceFunctions.FLOAT_BRAY_CURTIS_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_128_BRAY_CURTIS_DISTANCE));
        pairs.add(new Pair<>("bray curtis float 256", DistanceFunctions.FLOAT_BRAY_CURTIS_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_256_BRAY_CURTIS_DISTANCE));

        pairs.add(new Pair<>("canberra float 128", DistanceFunctions.FLOAT_CANBERRA_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_128_CANBERRA_DISTANCE));
        pairs.add(new Pair<>("canberra float 256", DistanceFunctions.FLOAT_CANBERRA_DISTANCE, Jdk17DistanceFunctions.VECTOR_FLOAT_256_CANBERRA_DISTANCE));

        for (Pair<DistanceFunction<float[], Float>, DistanceFunction<float[], Float>> pair : pairs) {

            long timeA = time(floatVector1, floatVector2, pair.a, numCompares);
            long timeB = time(floatVector1, floatVector2, pair.b, numCompares);

            System.out.printf("%s - a: %d, b: %d, single core improvement: %d %n", pair.name, timeA, timeB, timeA  - timeB);

        }
    }

    private <TVector, TDistance> long time(TVector floatVector1,
                                           TVector floatVector2,
                                           DistanceFunction<TVector, TDistance> function,
                                           int numOps) {
        long start = System.currentTimeMillis();

        for (int i = 0; i < numOps; i++) {
            function.distance(floatVector1, floatVector2);
        }
        long end = System.currentTimeMillis();
        return end - start;
    }

    private static float[] createRandomVector(int size) {
        return createRandomVector(size, System.nanoTime());
    }

    private static float[] createRandomVector(int size, long seed) {
        Random random = new Random(seed);

        float[] result = new float[size];

        for (int i = 0; i < size; i++) {
            result[i] = 1 - random.nextFloat(2f);
        }

        return result;
    }

    static class Pair<A, B> {
        String name;
        A a;
        B b;

        Pair(String name, A a, B b) {
            this.name = name;
            this.a = a;
            this.b = b;
        }
    }
}
