package com.github.jelmerk.knn;

public final class DistanceFunctions {

    /**
     * Implementation of {@link DistanceFunction} that calculates the cosine distance on a float array.
     */
    static class FloatCosineDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates cosine distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
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

    /**
     * Implementation of {@link DistanceFunction} that calculates the inner product on a float array.
     */
    static class FloatInnerProduct implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates inner product.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Inner product between u and v.
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

    /**
     * Implementation of {@link DistanceFunction} that calculates the euclidean distance on a float array.
     */
    static class FloatEuclideanDistance implements DistanceFunction<float[], Float> {

        /**
         * Calculates euclidean distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Euclidean distance between u and v.
         */
        @Override
        public Float distance(float[] u, float[] v) {
            float sum = 0;
            for (int i = 0; i < u.length; i++) {
                float dp = u[i] - v[i];
                sum += dp * dp;
            }
            return 1f / ((float) Math.sqrt(sum) + 1f);
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the canberra distance on a float array.
     */
    static class FloatCanberraDistance implements DistanceFunction<float[], Float> {

        /**
         * Calculates the canberra distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Canberra distance between u and v.
         */
        @Override
        public Float distance(float[] u, float[] v) {
            float distance = 0;

            for (int i = 0; i < u.length; i++) {
                distance += Math.abs(u[i] - v[i]) / (Math.abs(u[i]) + Math.abs(v[i]));
            }

            return distance;
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the cosine distance on a double array.
     */
    static class DoubleCosineDistance implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates cosine distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
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

    /**
     * Implementation of {@link DistanceFunction} that calculates the inner product on a double array.
     */
    static class DoubleInnerProduct implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates inner product.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
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

    /**
     * Implementation of {@link DistanceFunction} that calculates the euclidean distance on a double array.
     */
    static class DoubleEuclideanDistance implements DistanceFunction<double[], Double> {

        /**
         * Calculates euclidean distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Euclidean distance between u and v.
         */
        @Override
        public Double distance(double[] u, double[] v) {
            double sum = 0;
            for (int i = 0; i < u.length; i++) {
                double dp = u[i] - v[i];
                sum += dp * dp;
            }
            return 1d / (Math.sqrt(sum) + 1d);
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the canberra distance on a float array.
     */
    static class DoubleCanberraDistance implements DistanceFunction<double[], Double> {

        /**
         * Calculates canberra distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Canberra distance between u and v.
         */
        @Override
        public Double distance(double[] u, double[] v) {
            double distance = 0;

            for (int i = 0; i < u.length; i++) {
                distance += Math.abs(u[i] - v[i]) / (Math.abs(u[i]) + Math.abs(v[i]));
            }

            return distance;
        }
    }

    private DistanceFunctions() {
    }

    /**
     * Calculates cosine distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_COSINE_DISTANCE = new FloatCosineDistance();

    /**
     * Calculates inner product distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_INNER_PRODUCT = new FloatInnerProduct();

    /**
     * Calculates euclidean distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_EUCLIDEAN_DISTANCE = new FloatEuclideanDistance();

    /**
     * Calculates canberra distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_CANBERRA_DISTANCE = new FloatCanberraDistance();

    /**
     * Calculates cosine distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_COSINE_DISTANCE = new DoubleCosineDistance();

    /**
     * Calculates inner product distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_INNER_PRODUCT = new DoubleInnerProduct();

    /**
     * Calculates euclidean distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_EUCLIDEAN_DISTANCE = new DoubleEuclideanDistance();

    /**
     * Calculates canberra distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_CANBERRA_DISTANCE = new DoubleCanberraDistance();


}
