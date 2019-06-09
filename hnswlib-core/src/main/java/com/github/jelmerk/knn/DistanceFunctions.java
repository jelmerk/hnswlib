package com.github.jelmerk.knn;

/**
 * Various distance metrics.
 */
public class DistanceFunctions {

    /**
     * Calculates cosine distance on a float array.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    @SuppressWarnings("Duplicates")
    public static float floatArrayCosineDistance(float[] u, float[] v)  {
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
     * Calculates cosine distance on a float array.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    @SuppressWarnings("Duplicates")
    public static double doubleArrayCosineDistance(double[] u, double[] v)  {
        double dot = 0.0f;
        double nru = 0.0f;
        double nrv = 0.0f;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
            nru += u[i] * u[i];
            nrv += v[i] * v[i];
        }

        double similarity = dot / (Math.sqrt(nru) * Math.sqrt(nrv));
        return 1 - similarity;
    }

    /**
     * Calculates inner product.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    @SuppressWarnings("Duplicates")
    public static float floatArrayInnerProduct(float[] u, float[] v) {
        float dot = 0;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
        }

        return 1 - dot;
    }

    /**
     * Calculates inner product.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    @SuppressWarnings("Duplicates")
    public static double doubleArrayInnerProduct(double[] u, double[] v) {
        double dot = 0;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
        }

        return 1 - dot;
    }

}
