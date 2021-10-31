package com.github.jelmerk.knn;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

public class Jdk17DistanceFunctionsTest {

    private double error = 1e-4;

    @Test
    void float128CosineDistance() {
        float[] floatVector1 = createRandomVector(128);
        float[] floatVector2 = createRandomVector(128);

        float result1 = DistanceFunctions.FLOAT_COSINE_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_COSINE_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float256CosineDistance() {
        float[] floatVector1 = createRandomVector(256);
        float[] floatVector2 = createRandomVector(256);

        float result1 = DistanceFunctions.FLOAT_COSINE_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_COSINE_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));

//        long time1 = time(floatVector1, floatVector2, DistanceFunctions.FLOAT_COSINE_DISTANCE, 100_000_000);
//        System.out.println(time1);
//        long time2 = time(floatVector1, floatVector2, Jdk17DistanceFunctions.VECTOR_FLOAT_256_COSINE_DISTANCE, 100_000_000);
//        System.out.println(time2);
    }

    @Test
    void float128InnerProduct() {
        float[] floatVector1 = createRandomVector(128);
        float[] floatVector2 = createRandomVector(128);

        float result1 = DistanceFunctions.FLOAT_INNER_PRODUCT.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_INNER_PRODUCT.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));

//        long time1 = time(floatVector1, floatVector2, DistanceFunctions.FLOAT_INNER_PRODUCT, 100_000_000);
//        System.out.println(time1);
//        long time2 = time(floatVector1, floatVector2, Jdk17DistanceFunctions.VECTOR_FLOAT_128_INNER_PRODUCT, 100_000_000);
//        System.out.println(time2);
    }

    @Test
    void float256InnerProduct() {
        float[] floatVector1 = createRandomVector(256);
        float[] floatVector2 = createRandomVector(256);

        float result1 = DistanceFunctions.FLOAT_INNER_PRODUCT.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_INNER_PRODUCT.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
    }

    @Test
    void float128EuclideanDistance() {
        float[] floatVector1 = createRandomVector(128);
        float[] floatVector2 = createRandomVector(128);

        float result1 = DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));

//        long time1 = time(floatVector1, floatVector2, DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE, 100_000_000);
//        System.out.println(time1);
//
//        long time2 = time(floatVector1, floatVector2, Jdk17DistanceFunctions.VECTOR_FLOAT_128_EUCLIDEAN_DISTANCE, 100_000_000);
//        System.out.println(time2);
    }

    @Test
    void float256EuclideanDistance() {
        float[] floatVector1 = createRandomVector(256);
        float[] floatVector2 = createRandomVector(256);

        float result1 = DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));
//
//        long time1 = time(floatVector1, floatVector2, DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE, 100_000_000);
//        System.out.println(time1);
//
//        long time2 = time(floatVector1, floatVector2, Jdk17DistanceFunctions.VECTOR_FLOAT_256_EUCLIDEAN_DISTANCE, 100_000_000);
//        System.out.println(time2);
    }


    @Test
    void float128ManhattanDistance() {
        float[] floatVector1 = createRandomVector(128, 123);
        float[] floatVector2 = createRandomVector(128, 456);

        float result1 = DistanceFunctions.FLOAT_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_128_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));

//        long time1 = time(floatVector1, floatVector2, DistanceFunctions.FLOAT_MANHATTAN_DISTANCE, 100_000_000);
//        System.out.println(time1);
//
//        long time2 = time(floatVector1, floatVector2, Jdk17DistanceFunctions.VECTOR_FLOAT_128_MANHATTAN_DISTANCE, 100_000_000);
//        System.out.println(time2);
    }

    @Test
    void float256ManhattanDistance() {
        float[] floatVector1 = createRandomVector(256);
        float[] floatVector2 = createRandomVector(256);

        float result1 = DistanceFunctions.FLOAT_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);
        float result2 = Jdk17DistanceFunctions.VECTOR_FLOAT_256_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2);

        assertThat((double) result2 , closeTo(result1, error));

//        long time1 = time(floatVector1, floatVector2, DistanceFunctions.FLOAT_MANHATTAN_DISTANCE, 100_000_000);
//        System.out.println(time1);
//
//        long time2 = time(floatVector1, floatVector2, Jdk17DistanceFunctions.VECTOR_FLOAT_128_MANHATTAN_DISTANCE, 100_000_000);
//        System.out.println(time2);
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
            result[i] = random.nextFloat(1f);
        }

        return result;
    }
}
