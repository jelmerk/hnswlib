package org.github.jelmerk.hnsw;

import java.util.List;

/**
 * Utilities to work with vectors.
 */
public final class VectorUtils {

    private VectorUtils() {
    }

    /**
     * Calculates magnitude of the vector.
     *
     * @param vector The vector to calculate magnitude for.
     * @return The magnitude.
     */
    public static float magnitude(List<Float> vector) {
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
    public static void normalize(List<Float> vector) {
        float normFactor = 1 / magnitude(vector);
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, vector.get(i) * normFactor);
        }
    }

}
