package com.github.jelmerk.knn;

import jdk.incubator.vector.*;

/**
 * Collection of distance functions only available on JDK 17.
 */
public final class Jdk17DistanceFunctions {

    static class VectorFloat128InnerProduct implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);

            for (int i = 0; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                vecSum = vecX.fma(vecY, vecSum);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);

            return 1 - dot;
        }
    }

    static class VectorFloat256InnerProduct implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);

            for (int i = 0; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                vecSum = vecX.fma(vecY, vecSum);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);

            return 1 - dot;
        }
    }

    static class VectorFloat128CosineDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);
            FloatVector xSquareV = FloatVector.zero(SPECIES_FLOAT_128);
            FloatVector ySquareV = FloatVector.zero(SPECIES_FLOAT_128);;

            for (int i = 0; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                vecSum = vecX.fma(vecY, vecSum);
                xSquareV = vecX.fma(vecX, xSquareV);
                ySquareV = vecY.fma(vecY, ySquareV);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);
            float nrv = ySquareV.reduceLanes(VectorOperators.ADD);
            float nru = xSquareV.reduceLanes(VectorOperators.ADD);

            float similarity = dot / (float)(Math.sqrt(nru) * Math.sqrt(nrv));
            return 1 - similarity;
        }
    }

    static class VectorFloat256CosineDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);
            FloatVector xSquareV = FloatVector.zero(SPECIES_FLOAT_256);
            FloatVector ySquareV = FloatVector.zero(SPECIES_FLOAT_256);;

            for (int i = 0; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                vecSum = vecX.fma(vecY, vecSum);
                xSquareV = vecX.fma(vecX, xSquareV);
                ySquareV = vecY.fma(vecY, ySquareV);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);
            float nrv = ySquareV.reduceLanes(VectorOperators.ADD);
            float nru = xSquareV.reduceLanes(VectorOperators.ADD);

            float similarity = dot / (float)(Math.sqrt(nru) * Math.sqrt(nrv));
            return 1 - similarity;
        }
    }


    static class VectorFloat128EuclideanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);
            for (int i = 0; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecDiff = vecX.sub(vecY);
                vecSum = vecDiff.fma(vecDiff, vecSum);
            }
            float sum = vecSum.reduceLanes(VectorOperators.ADD);
            return (float) Math.sqrt(sum);
        }
    }

    static class VectorFloat256EuclideanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);
            for (int i = 0; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecDiff = vecX.sub(vecY);
                vecSum = vecDiff.fma(vecDiff, vecSum);
            }
            float sum = vecSum.reduceLanes(VectorOperators.ADD);
            return (float) Math.sqrt(sum);
        }
    }


    static class VectorFloat128ManhattanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);
            for (int i = 0; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                FloatVector vecDiff = vecX.sub(vecY).abs();
                vecSum = vecSum.add(vecDiff);
            }
            return vecSum.abs().reduceLanes(VectorOperators.ADD);
        }
    }

    static class VectorFloat256ManhattanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);
            for (int i = 0; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecX = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecY = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                FloatVector vecDiff = vecX.sub(vecY).abs();
                vecSum = vecSum.add(vecDiff);
            }
            return vecSum.abs().reduceLanes(VectorOperators.ADD);
        }
    }

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_COSINE_DISTANCE = new VectorFloat128CosineDistance();

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_COSINE_DISTANCE = new VectorFloat256CosineDistance();

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_INNER_PRODUCT = new VectorFloat128InnerProduct();

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_INNER_PRODUCT = new VectorFloat256InnerProduct();

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_EUCLIDEAN_DISTANCE = new VectorFloat128EuclideanDistance();

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_EUCLIDEAN_DISTANCE = new VectorFloat256EuclideanDistance();

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_MANHATTAN_DISTANCE = new VectorFloat128ManhattanDistance();

    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_MANHATTAN_DISTANCE = new VectorFloat256ManhattanDistance();

}
