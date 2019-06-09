package com.github.jelmer.knn.spark;

import org.apache.spark.ml.linalg.DenseVector;

public class DistanceFunctions {

    /**
     * Calculates cosine distance.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    public static double cosineDistance(DenseVector u, DenseVector v) {
        if (u.size() != v.size()) {
            throw new IllegalArgumentException("Vectors have non-matching dimensions");
        }

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
     * Calculates inner product.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    public static double innerProduct(DenseVector u, DenseVector v) {
        if (u.size() != v.size()) {
            throw new IllegalArgumentException("Vectors have non-matching dimensions");
        }

        double dot = 0;
        for (int i = 0; i < u.size(); i++) {
            dot += u.apply(i) * v.apply(i);
        }

        return 1 - dot;
    }
}
