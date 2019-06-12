package com.github.jelmerk.knn.util;

public final class VectorUtils {

    private VectorUtils() {
    }
    /**
     * Calculates the magnitude of the vector.
     *
     * @param vector The vector to calculate magnitude for.
     * @return The magnitude.
     */
    public static float magnitude(float[] vector) {
        float magnitude = 0.0f;
        for (Float aFloat : vector) {
            magnitude += aFloat * aFloat;
        }
        return (float) Math.sqrt(magnitude);
    }

    /**
     * Turns vector to unit vector.
     *
     * @param vector The vector to normalize.
     */
    public static float[] normalize(float[] vector) {

        float[] result = new float[vector.length];

        float normFactor = 1 / magnitude(vector);
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * normFactor;
        }
        return result;
    }

}