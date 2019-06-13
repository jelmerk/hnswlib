package com.github.jelmer.knn.spark;

import org.apache.spark.ml.linalg.Vector;

public final class DenseVectorDistanceFunctions {

    /**
     * Private constructor to prevent initialization.
     */
    private DenseVectorDistanceFunctions() {
    }

    /**
     * Calculates cosine distance on a dense vector.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    @SuppressWarnings("Duplicates")
    public static double cosineDistance(Vector u, Vector v) {
        double dot = 0.0f;
        double nru = 0.0f;
        double nrv = 0.0f;
        for (int i = 0; i < u.size(); i++) {
            dot += u.apply(i) * v.apply(i);
            nru += u.apply(i) * u.apply(i);
            nrv += v.apply(i) * v.apply(i);
        }

        double similarity = dot / (Math.sqrt(nru) * Math.sqrt(nrv));
        return 1 - similarity;
    }


    /**
     * Calculates inner product on a dense vector.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    @SuppressWarnings("Duplicates")
    public static double innerProduct(Vector u, Vector v) {
        double dot = 0;
        for (int i = 0; i < u.size(); i++) {
            dot += u.apply(i) * v.apply(i);
        }

        return 1 - dot;
    }


}
