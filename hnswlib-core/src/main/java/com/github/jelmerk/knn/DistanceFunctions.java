package com.github.jelmerk.knn;

public final class DistanceFunctions {

    /**
     * Implementation of {@link DistanceFunction} that calculates the inner product.
     */
    static class FloatSparseVectorInnerProduct implements DistanceFunction<SparseVector<float[]>, Float> {

        /**
         * Calculates the inner product.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Inner product between u and v.
         */
        @Override
        public Float distance(SparseVector<float[]> u, SparseVector<float[]> v) {
            int[] uIndices = u.indices();
            float[] uValues = u.values();
            int[] vIndices = v.indices();
            float[] vValues = v.values();
            float dot = 0.0f;
            int i = 0;
            int j = 0;

            while (i < uIndices.length && j < vIndices.length) {
                if (uIndices[i] < vIndices[j]) {
                    i += 1;
                } else if (uIndices[i] > vIndices[j]) {
                    j += 1;
                } else {
                    dot += uValues[i] * vValues[j];
                    i += 1;
                    j += 1;
                }
            }
            return 1 - dot;
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the inner product.
     */
    static class DoubleSparseVectorInnerProduct implements DistanceFunction<SparseVector<double[]>, Double> {

        /**
         * Calculates the inner product.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Inner product between u and v.
         */
        @Override
        public Double distance(SparseVector<double[]> u, SparseVector<double[]> v) {
            int[] uIndices = u.indices();
            double[] uValues = u.values();
            int[] vIndices = v.indices();
            double[] vValues = v.values();
            double dot = 0.0f;
            int i = 0;
            int j = 0;

            while (i < uIndices.length && j < vIndices.length) {
                if (uIndices[i] < vIndices[j]) {
                    i += 1;
                } else if (uIndices[i] > vIndices[j]) {
                    j += 1;
                } else {
                    dot += uValues[i] * vValues[j];
                    i += 1;
                    j += 1;
                }
            }
            return 1 - dot;
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the cosine distance.
     */
    static class FloatCosineDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the cosine distance.
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
     * Implementation of {@link DistanceFunction} that calculates the inner product.
     */
    static class FloatInnerProduct implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the inner product.
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
     * Implementation of {@link DistanceFunction} that calculates the euclidean distance.
     */
    static class FloatEuclideanDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the euclidean distance.
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
            return (float) Math.sqrt(sum);
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the canberra distance.
     */
    static class FloatCanberraDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

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
            float sum = 0;
            for (int i = 0; i < u.length; i++) {
                float num = Math.abs(u[i] - v[i]);
                float denom = Math.abs(u[i]) + Math.abs(v[i]);
                sum += num == 0.0 && denom == 0.0 ? 0.0 : num / denom;
            }
            return sum;
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the Bray Curtis distance.
     */
    static class FloatBrayCurtisDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the Bray Curtis distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Bray Curtis distance between u and v.
         */
        @Override
        public Float distance(float[] u, float[] v) {

            float sump = 0;
            float sumn = 0;

            for (int i = 0; i < u.length; i++) {
                sumn += Math.abs(u[i] - v[i]);
                sump += Math.abs(u[i] + v[i]);
            }

            return sumn / sump;
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the correlation distance.
     */
    static class FloatCorrelationDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the correlation distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Correlation distance between u and v.
         */
        @Override
        public Float distance(float[] u, float[] v) {
            float x = 0;
            float y = 0;

            for (int i = 0; i < u.length; i++) {
                x += -u[i];
                y += -v[i];
            }

            x /= u.length;
            y /= v.length;

            float num = 0;
            float den1 = 0;
            float den2 = 0;
            for (int i = 0; i < u.length; i++) {
                num += (u[i] + x) * (v[i] + y);

                den1 += Math.abs(Math.pow(u[i] + x, 2));
                den2 += Math.abs(Math.pow(v[i] + x, 2));
            }

            return 1f - (num / ((float) Math.sqrt(den1) * (float) Math.sqrt(den2)));
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the manhattan distance.
     */
    static class FloatManhattanDistance implements DistanceFunction<float[], Float> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the manhattan distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Manhattan distance between u and v.
         */
        @Override
        public Float distance(float[] u, float[] v) {
            float sum = 0;
            for (int i = 0; i < u.length; i++) {
                sum += Math.abs(u[i] - v[i]);
            }
            return sum;
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the cosine distance.
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
     * Implementation of {@link DistanceFunction} that calculates the inner product.
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
     * Implementation of {@link DistanceFunction} that calculates the euclidean distance.
     */
    static class DoubleEuclideanDistance implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

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
            return Math.sqrt(sum);
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the canberra distance.
     */
    static class DoubleCanberraDistance implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

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
            double sum = 0;
            for (int i = 0; i < u.length; i++) {
                double num = Math.abs(u[i] - v[i]);
                double denom = Math.abs(u[i]) + Math.abs(v[i]);
                sum += num == 0.0 && denom == 0.0 ? 0.0 : num / denom;
            }
            return sum;
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the Bray Curtis distance.
     */
    static class DoubleBrayCurtisDistance implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the Bray Curtis distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Bray Curtis distance between u and v.
         */
        @Override
        public Double distance(double[] u, double[] v) {
            double sump = 0;
            double sumn = 0;

            for (int i = 0; i < u.length; i++) {
                sumn += Math.abs(u[i] - v[i]);
                sump += Math.abs(u[i] + v[i]);
            }

            return sumn / sump;
        }
    }


    /**
     * Implementation of {@link DistanceFunction} that calculates the correlation distance.
     */
    static class DoubleCorrelationDistance implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the correlation distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Correlation distance between u and v.
         */
        @Override
        public Double distance(double[] u, double[] v) {
            double x = 0;
            double y = 0;

            for (int i = 0; i < u.length; i++) {
                x += -u[i];
                y += -v[i];
            }

            x /= u.length;
            y /= v.length;

            double num = 0;
            double den1 = 0;
            double den2 = 0;
            for (int i = 0; i < u.length; i++) {
                num += (u[i] + x) * (v[i] + y);

                den1 += Math.abs(Math.pow(u[i] + x, 2));
                den2 += Math.abs(Math.pow(v[i] + x, 2));
            }

            return 1 - (num / (Math.sqrt(den1) * Math.sqrt(den2)));
        }
    }

    /**
     * Implementation of {@link DistanceFunction} that calculates the manhattan distance.
     */
    static class DoubleManhattanDistance implements DistanceFunction<double[], Double> {

        private static final long serialVersionUID = 1L;

        /**
         * Calculates the manhattan distance.
         *
         * @param u Left vector.
         * @param v Right vector.
         *
         * @return Manhattan distance between u and v.
         */
        @Override
        public Double distance(double[] u, double[] v) {
            double sum = 0;
            for (int i = 0; i < u.length; i++) {
                sum += Math.abs(u[i] - v[i]);
            }
            return sum;
        }
    }

    private DistanceFunctions() {
    }

    /**
     * Calculates the cosine distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_COSINE_DISTANCE = new FloatCosineDistance();

    /**
     * Calculates the inner product distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_INNER_PRODUCT = new FloatInnerProduct();

    /**
     * Calculates the euclidean distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_EUCLIDEAN_DISTANCE = new FloatEuclideanDistance();

    /**
     * Calculates the canberra distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_CANBERRA_DISTANCE = new FloatCanberraDistance();

    /**
     * Calculates the bray curtis distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_BRAY_CURTIS_DISTANCE = new FloatBrayCurtisDistance();

    /**
     * Calculates the correlation distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_CORRELATION_DISTANCE = new FloatCorrelationDistance();

    /**
     * Calculates the manhattan distance.
     */
    public static final DistanceFunction<float[], Float> FLOAT_MANHATTAN_DISTANCE = new FloatManhattanDistance();

    /**
     * Calculates the cosine distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_COSINE_DISTANCE = new DoubleCosineDistance();

    /**
     * Calculates the inner product.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_INNER_PRODUCT = new DoubleInnerProduct();

    /**
     * Calculates the euclidean distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_EUCLIDEAN_DISTANCE = new DoubleEuclideanDistance();

    /**
     * Calculates the canberra distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_CANBERRA_DISTANCE = new DoubleCanberraDistance();

    /**
     * Calculates the bray curtis distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_BRAY_CURTIS_DISTANCE = new DoubleBrayCurtisDistance();

    /**
     * Calculates the correlation distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_CORRELATION_DISTANCE = new DoubleCorrelationDistance();

    /**
     * Calculates the manhattan distance.
     */
    public static final DistanceFunction<double[], Double> DOUBLE_MANHATTAN_DISTANCE = new DoubleManhattanDistance();

    /**
     * Calculates the inner product.
     */
    public static final DistanceFunction<SparseVector<float[]>, Float> FLOAT_SPARSE_VECTOR_INNER_PRODUCT =
            new FloatSparseVectorInnerProduct();

    /**
     * Calculates the inner product.
     */
    public static final DistanceFunction<SparseVector<double[]>, Double> DOUBLE_SPARSE_VECTOR_INNER_PRODUCT =
            new DoubleSparseVectorInnerProduct();
}
