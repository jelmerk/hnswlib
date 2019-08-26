package com.github.jelmerk.knn.spark;

import com.github.jelmerk.knn.DistanceFunction;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.Vector;

public class DistanceFunctions {

    static class DenseVectorCosineDistance implements DistanceFunction<Vector, Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates cosine distance on a dense vector.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Double distance(Vector u, Vector v) {
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
    }

    static class DenseVectorInnerProductDistance implements DistanceFunction<Vector, Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates inner product on a dense vector.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Double distance(Vector u, Vector v) {
            double dot = 0;
            for (int i = 0; i < u.size(); i++) {
                dot += u.apply(i) * v.apply(i);
            }

            return 1 - dot;
        }
    }

    static class SparseVectorCosineDistance implements DistanceFunction<Vector, Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates cosine distance on a sparse vector.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Double distance(Vector u, Vector v) {
            double dot = 0.0f;
            double nru = 0.0f;
            double nrv = 0.0f;

            for (int i : ((SparseVector)u).indices()) {
                dot += u.apply(i) * v.apply(i);
                nru += u.apply(i) * u.apply(i);
                nrv += v.apply(i) * v.apply(i);
            }

            double similarity = dot / (Math.sqrt(nru) * Math.sqrt(nrv));
            return 1 - similarity;
        }
    }

    static class SparseVectorInnerProductDistance implements DistanceFunction<Vector, Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates inner product on a sparse vector.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Double distance(Vector u, Vector v) {
            double dot = 0;
            for (int i : ((SparseVector)u).indices()) {
                dot += u.apply(i) * v.apply(i);
            }
            return 1 - dot;
        }
    }

    private DistanceFunctions() {
    }

    /**
     * Calculates cosine distance on a dense vector.
     */
    public static final DistanceFunction<Vector, Double> DENSE_VECTOR_COSINE_DISTANCE = new DenseVectorCosineDistance();

    /**
     * Calculates inner product distance on a dense vector.
     */
    public static final DistanceFunction<Vector, Double> DENSE_VECTOR_INNER_PRODUCT = new DenseVectorInnerProductDistance();

    /**
     * Calculates cosine distance on a sparse vector.
     */
    public static final DistanceFunction<Vector, Double> SPARSE_VECTOR_COSINE_DISTANCE = new SparseVectorCosineDistance();

    /**
     * Calculates inner product distance on sparse vector.
     */
    public static final DistanceFunction<Vector, Double> SPARSE_VECTOR_INNER_PRODUCT = new SparseVectorInnerProductDistance();

}
