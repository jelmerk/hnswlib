package com.github.jelmerk.knn;

/**
 * Various distance metrics.
 */
public class DistanceFunctions {

    /**
     * Calculates cosine distance.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    public static float cosineDistance(float[] u, float[] v)  {
        float dot = 0.0f;
        float nru = 0.0f;
        float nrv = 0.0f;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
            nru += u[i] * u[i];
            nrv += v[i] * v[i];
        }

        float similarity = dot / (float)(Math.sqrt(nru) * Math.sqrt(nrv));
        return 1 - similarity;
    }

    /**
     * Calculates inner product.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    public static float innerProduct(float[] u, float[] v) {
        float dot = 0;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
        }

        return 1 - dot;
    }

}
