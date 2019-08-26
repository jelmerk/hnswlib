package com.github.jelmerk.knn;

public final class DistanceFunctions {

    static class FloatCosineDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates cosine distance on a float array.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Float distance(float[] u, float[] v) {
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
    }

    static class FloatInnerProduct implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates inner product.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Float distance(float[] u, float[] v) {
            float dot = 0;
            for (int i = 0; i < u.length; i++) {
                dot += u[i] * v[i];
            }

            return 1 - dot;
        }
    }

    static class DoubleCosineDistance implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates cosine distance on a float array.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Double distance(double[] u, double[] v) {
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
    }

    static class DoubleInnerProduct implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates inner product.
         *
         * @param u Left vector.
         * @param v Right vector.
         * @return Cosine distance between u and v.
         */
        @Override
        public Double distance(double[] u, double[] v) {
            double dot = 0;
            for (int i = 0; i < u.length; i++) {
                dot += u[i] * v[i];
            }

            return 1 - dot;
        }
    }

    private DistanceFunctions() {
    }

    /**
     * Calculates cosine distance on a float array.
     */
    public static final DistanceFunction<float[], Float> FLOAT_COSINE_DISTANCE = new FloatCosineDistance();

    /**
     * Calculates inner product distance on a float array.
     */
    public static final DistanceFunction<float[], Float> FLOAT_INNER_PRODUCT = new FloatInnerProduct();

    /**
     * Calculates cosine distance on a float array.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_COSINE_DISTANCE = new DoubleCosineDistance();

    /**
     * Calculates inner product distance on a float array.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_INNER_PRODUCT = new DoubleInnerProduct();

}
