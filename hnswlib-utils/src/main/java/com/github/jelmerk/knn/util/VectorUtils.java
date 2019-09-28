package com.github.jelmerk.knn.util;

/**
 * Misc utility methods for dealing with vectors.
 */
public final class VectorUtils {

    private VectorUtils() {
    }

    /**
     * Calculates the magnitude of the vector.
     *
     * @param vector The vector to calculate magnitude for.
     * @return The magnitude.
     */
    public static double magnitude(double[] vector) {
        double magnitude = 0.0f;
        for (double aDouble : vector) {
            magnitude += aDouble * aDouble;
        }
        return Math.sqrt(magnitude);
    }

    /**
     * Turns vector to unit vector.
     *
     * @param vector The vector to normalize.
     * @return the input vector as a unit vector
     */
    public static double[] normalize(double[] vector) {

        double[] result = new double[vector.length];

        double normFactor = 1 / magnitude(vector);
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * normFactor;
        }
        return result;
    }

    /**
     * Calculates the magnitude of the vector.
     *
     * @param vector The vector to calculate magnitude for.
     * @return The magnitude.
     */
    public static float magnitude(float[] vector) {
        float magnitude = 0.0f;
        for (float aFloat : vector) {
            magnitude += aFloat * aFloat;
        }
        return (float) Math.sqrt(magnitude);
    }

    /**
     * Turns vector to unit vector.
     *
     * @param vector The vector to normalize.
     * @return the input vector as a unit vector
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