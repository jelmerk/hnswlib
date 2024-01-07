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

            int i = 0;
            for (; (i + SPECIES_FLOAT_128.length()) <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                vecSum = vecU.fma(vecV, vecSum);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                dot += u[i] * v[i];
            }
            return 1 - dot;
        }
    }

    static class VectorFloat256InnerProduct implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);

            int i = 0;
            for (; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                vecSum = vecU.fma(vecV, vecSum);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                dot += u[i] * v[i];
            }
            return 1 - dot;
        }
    }

    static class VectorFloat128CosineDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);
            FloatVector xSquareV = FloatVector.zero(SPECIES_FLOAT_128);
            FloatVector ySquareV = FloatVector.zero(SPECIES_FLOAT_128);

            int i = 0;
            for (; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                vecSum = vecU.fma(vecV, vecSum);
                xSquareV = vecU.fma(vecU, xSquareV);
                ySquareV = vecV.fma(vecV, ySquareV);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);
            float nrv = ySquareV.reduceLanes(VectorOperators.ADD);
            float nru = xSquareV.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                dot += u[i] * v[i];
                nru += u[i] * u[i];
                nrv += v[i] * v[i];
            }

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
            FloatVector ySquareV = FloatVector.zero(SPECIES_FLOAT_256);

            int i = 0;
            for (; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                vecSum = vecU.fma(vecV, vecSum);
                xSquareV = vecU.fma(vecU, xSquareV);
                ySquareV = vecV.fma(vecV, ySquareV);
            }
            float dot = vecSum.reduceLanes(VectorOperators.ADD);
            float nrv = ySquareV.reduceLanes(VectorOperators.ADD);
            float nru = xSquareV.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                dot += u[i] * v[i];
                nru += u[i] * u[i];
                nrv += v[i] * v[i];
            }

            float similarity = dot / (float)(Math.sqrt(nru) * Math.sqrt(nrv));
            return 1 - similarity;
        }
    }

    static class VectorFloat128EuclideanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);

            int i = 0;
            for (; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecDiff = vecU.sub(vecV);
                vecSum = vecDiff.fma(vecDiff, vecSum);
            }
            float sum = vecSum.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                float dp = u[i] - v[i];
                sum += dp * dp;
            }

            return (float) Math.sqrt(sum);
        }
    }

    static class VectorFloat256EuclideanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);
            int i = 0;
            for (; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecDiff = vecU.sub(vecV);
                vecSum = vecDiff.fma(vecDiff, vecSum);
            }
            float sum = vecSum.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                float dp = u[i] - v[i];
                sum += dp * dp;
            }

            return (float) Math.sqrt(sum);
        }
    }


    static class VectorFloat128ManhattanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);
            int i = 0;
            for (; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                FloatVector vecDiff = vecU.sub(vecV).abs();
                vecSum = vecSum.add(vecDiff);
            }

            float sum = vecSum.reduceLanes(VectorOperators.ADD);
            for (; i < u.length; i++) {
                sum += Math.abs(u[i] - v[i]);
            }

            return sum;
        }
    }

    static class VectorFloat256ManhattanDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);
            int i = 0;
            for (; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                FloatVector vecDiff = vecU.sub(vecV).abs();
                vecSum = vecSum.add(vecDiff);
            }

            float sum = vecSum.reduceLanes(VectorOperators.ADD);
            for (; i < u.length; i++) {
                sum += Math.abs(u[i] - v[i]);
            }

            return sum;
        }
    }

    static class VectorFloat128CanberraDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {

            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_128);
            int i = 0;
            for (; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);
                FloatVector num = vecU.sub(vecV).abs();
                FloatVector denom = vecU.abs().add(vecV.abs());
                vecSum = vecSum.add(num.div(denom));
            }

            float sum = vecSum.reduceLanes(VectorOperators.ADD);
            for (; i < u.length; i++) {
                double num = Math.abs(u[i] - v[i]);
                double denominator = Math.abs(u[i]) + Math.abs(v[i]);
                sum += (float) (num == 0.0 && denominator == 0.0 ? 0.0 : num / denominator);
            }

            return sum;
        }
    }

    static class VectorFloat256CanberraDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSum = FloatVector.zero(SPECIES_FLOAT_256);

            int i = 0;
            for (; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);
                FloatVector num = vecU.sub(vecV).abs();
                FloatVector denominator = vecU.abs().add(vecV.abs());
                vecSum = vecSum.add(num.div(denominator));
            }
            float sum = vecSum.reduceLanes(VectorOperators.ADD);
            for (; i < u.length; i++) {
                double num = Math.abs(u[i] - v[i]);
                double denominator = Math.abs(u[i]) + Math.abs(v[i]);
                sum += (float) (num == 0.0 && denominator == 0.0 ? 0.0 : num / denominator);
            }

            return sum;
        }
    }

    static class VectorFloat128BrayCurtisDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_128 = FloatVector.SPECIES_128;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSump = FloatVector.zero(SPECIES_FLOAT_128);
            FloatVector vecSumn = FloatVector.zero(SPECIES_FLOAT_128);
            int i = 0;
            for (; i + SPECIES_FLOAT_128.length() <= u.length; i += SPECIES_FLOAT_128.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_128, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_128, v, i);

                vecSumn = vecSumn.add(vecU.sub(vecV).abs());
                vecSump = vecSump.add(vecU.add(vecV).abs());
            }

            float sumn = vecSumn.reduceLanes(VectorOperators.ADD);
            float sump = vecSump.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                sumn += Math.abs(u[i] - v[i]);
                sump += Math.abs(u[i] + v[i]);
            }

            return sumn / sump;
        }
    }

    static class VectorFloat256BrayCurtisDistance implements DistanceFunction<float[], Float> {

        private static final VectorSpecies<Float> SPECIES_FLOAT_256 = FloatVector.SPECIES_256;

        @Override
        public Float distance(float[] u, float[] v) {
            FloatVector vecSump = FloatVector.zero(SPECIES_FLOAT_256);
            FloatVector vecSumn = FloatVector.zero(SPECIES_FLOAT_256);

            int i = 0;
            for (; i + SPECIES_FLOAT_256.length() <= u.length; i += SPECIES_FLOAT_256.length()) {
                FloatVector vecU = FloatVector.fromArray(SPECIES_FLOAT_256, u, i);
                FloatVector vecV = FloatVector.fromArray(SPECIES_FLOAT_256, v, i);

                vecSumn = vecSumn.add(vecU.sub(vecV).abs());
                vecSump = vecSump.add(vecU.add(vecV).abs());
            }

            float sumn = vecSumn.reduceLanes(VectorOperators.ADD);
            float sump = vecSump.reduceLanes(VectorOperators.ADD);

            for (; i < u.length; i++) {
                sumn += Math.abs(u[i] - v[i]);
                sump += Math.abs(u[i] + v[i]);
            }

            return sumn / sump;
        }
    }

    /**
     * Calculates the cosine distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_COSINE_DISTANCE = new VectorFloat128CosineDistance();

    /**
     * Calculates the cosine distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_COSINE_DISTANCE = new VectorFloat256CosineDistance();

    /**
     * Calculates the inner product distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_INNER_PRODUCT = new VectorFloat128InnerProduct();

    /**
     * Calculates the inner product distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_INNER_PRODUCT = new VectorFloat256InnerProduct();

    /**
     * Calculates the euclidean distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_EUCLIDEAN_DISTANCE = new VectorFloat128EuclideanDistance();

    /**
     * Calculates the euclidean distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_EUCLIDEAN_DISTANCE = new VectorFloat256EuclideanDistance();

    /**
     * Calculates the manhattan distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_MANHATTAN_DISTANCE = new VectorFloat128ManhattanDistance();

    /**
     * Calculates the manhattan distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_MANHATTAN_DISTANCE = new VectorFloat256ManhattanDistance();

    /**
     * Calculates the canberra distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_CANBERRA_DISTANCE = new VectorFloat128CanberraDistance();

    /**
     * Calculates the canberra distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_CANBERRA_DISTANCE = new VectorFloat256CanberraDistance();

    /**
     * Calculates the bray curtis distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_128_BRAY_CURTIS_DISTANCE = new VectorFloat128BrayCurtisDistance();

    /**
     * Calculates the bray curtis distance.
     */
    public static final DistanceFunction<float[], Float> VECTOR_FLOAT_256_BRAY_CURTIS_DISTANCE = new VectorFloat256BrayCurtisDistance();


}
