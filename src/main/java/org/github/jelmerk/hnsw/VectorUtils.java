package org.github.jelmerk.hnsw;

import java.util.List;

public class VectorUtils {

    private VectorUtils() {
    }

    public static float magnitude(List<Float> vector) {
        float magnitude = 0.0f;
        for (Float aFloat : vector) {
            magnitude += aFloat * aFloat;
        }
        return (float) Math.sqrt(magnitude);
    }

    public static void normalize(List<Float> vector) {
        float normFactor = 1 / magnitude(vector);
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, vector.get(i) * normFactor);
        }
    }

}
