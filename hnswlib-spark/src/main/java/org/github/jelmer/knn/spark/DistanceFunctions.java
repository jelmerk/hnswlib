package org.github.jelmer.knn.spark;

import org.apache.spark.ml.linalg.Vector;

public class DistanceFunctions {

    public static double cosineDistance(Vector u, Vector v) {
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


    public static double innerProduct(Vector u, Vector v) {
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
