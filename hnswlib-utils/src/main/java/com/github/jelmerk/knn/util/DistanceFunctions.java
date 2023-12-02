package com.github.jelmerk.knn.util;

public final class DistanceFunctions {

    private DistanceFunctions() {
    }


}








//public final class DistanceFunctions {
//
//
//    static class FloatSparseVectorInnerProduct implements DistanceFunction<SparseVector<float[]>, Float> {
//
//        @Override
//        public Float distance(SparseVector<float[]> u, SparseVector<float[]> v) {
//            int[] uIndices = u.indices();
//            float[] uValues = u.values();
//            int[] vIndices = v.indices();
//            float[] vValues = v.values();
//            float dot = 0.0f;
//            int i = 0;
//            int j = 0;
//
//            while (i < uIndices.length && j < vIndices.length) {
//                if (uIndices[i] < vIndices[j]) {
//                    i += 1;
//                } else if (uIndices[i] > vIndices[j]) {
//                    j += 1;
//                } else {
//                    dot += uValues[i] * vValues[j];
//                    i += 1;
//                    j += 1;
//                }
//            }
//            return 1 - dot;
//        }
//    }
//
//
//    static class DoubleSparseVectorInnerProduct implements DistanceFunction<SparseVector<double[]>, Double> {
//
//
//        @Override
//        public Double distance(SparseVector<double[]> u, SparseVector<double[]> v) {
//            int[] uIndices = u.indices();
//            double[] uValues = u.values();
//            int[] vIndices = v.indices();
//            double[] vValues = v.values();
//            double dot = 0.0f;
//            int i = 0;
//            int j = 0;
//
//            while (i < uIndices.length && j < vIndices.length) {
//                if (uIndices[i] < vIndices[j]) {
//                    i += 1;
//                } else if (uIndices[i] > vIndices[j]) {
//                    j += 1;
//                } else {
//                    dot += uValues[i] * vValues[j];
//                    i += 1;
//                    j += 1;
//                }
//            }
//            return 1 - dot;
//        }
//    }

//    static class FloatCosineDistance implements DistanceFunction<float[], Float> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Float distance(float[] u, float[] v) {
//            float dot = 0.0f;
//            float nru = 0.0f;
//            float nrv = 0.0f;
//            for (int i = 0; i < u.length; i++) {
//                dot += u[i] * v[i];
//                nru += u[i] * u[i];
//                nrv += v[i] * v[i];
//            }
//
//            float similarity = dot / (float)(Math.sqrt(nru) * Math.sqrt(nrv));
//            return 1 - similarity;
//        }
//    }
//
//
//    static class FloatInnerProduct implements DistanceFunction<float[], Float> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Float distance(float[] u, float[] v) {
//            float dot = 0;
//            for (int i = 0; i < u.length; i++) {
//                dot += u[i] * v[i];
//            }
//
//            return 1 - dot;
//        }
//    }
//
//    static class FloatEuclideanDistance implements DistanceFunction<float[], Float> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Float distance(float[] u, float[] v) {
//            float sum = 0;
//            for (int i = 0; i < u.length; i++) {
//                float dp = u[i] - v[i];
//                sum += dp * dp;
//            }
//            return (float) Math.sqrt(sum);
//        }
//    }
//
//    static class FloatCanberraDistance implements DistanceFunction<float[], Float> {
//
//        private static final long serialVersionUID = 1L;
//
//        @Override
//        public Float distance(float[] u, float[] v) {
//            float sum = 0;
//            for (int i = 0; i < u.length; i++) {
//                float num = Math.abs(u[i] - v[i]);
//                float denom = Math.abs(u[i]) + Math.abs(v[i]);
//                sum += num == 0.0 && denom == 0.0 ? 0.0 : num / denom;
//            }
//            return sum;
//        }
//    }
//
//    static class FloatBrayCurtisDistance implements DistanceFunction<float[], Float> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Float distance(float[] u, float[] v) {
//
//            float sump = 0;
//            float sumn = 0;
//
//            for (int i = 0; i < u.length; i++) {
//                sumn += Math.abs(u[i] - v[i]);
//                sump += Math.abs(u[i] + v[i]);
//            }
//
//            return sumn / sump;
//        }
//    }
//
//
//    static class FloatCorrelationDistance implements DistanceFunction<float[], Float> {
//
//        private static final long serialVersionUID = 1L;
//
//        @Override
//        public Float distance(float[] u, float[] v) {
//            float x = 0;
//            float y = 0;
//
//            for (int i = 0; i < u.length; i++) {
//                x += -u[i];
//                y += -v[i];
//            }
//
//            x /= u.length;
//            y /= v.length;
//
//            float num = 0;
//            float den1 = 0;
//            float den2 = 0;
//            for (int i = 0; i < u.length; i++) {
//                num += (u[i] + x) * (v[i] + y);
//
//                den1 += Math.abs(Math.pow(u[i] + x, 2));
//                den2 += Math.abs(Math.pow(v[i] + x, 2));
//            }
//
//            return 1f - (num / ((float) Math.sqrt(den1) * (float) Math.sqrt(den2)));
//        }
//    }
//
//    static class FloatManhattanDistance implements DistanceFunction<float[], Float> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Float distance(float[] u, float[] v) {
//            float sum = 0;
//            for (int i = 0; i < u.length; i++) {
//                sum += Math.abs(u[i] - v[i]);
//            }
//            return sum;
//        }
//    }
//
//    static class DoubleCosineDistance implements DistanceFunction<double[], Double> {
//
//        private static final long serialVersionUID = 1L;
//
//        @Override
//        public Double distance(double[] u, double[] v) {
//            double dot = 0.0f;
//            double nru = 0.0f;
//            double nrv = 0.0f;
//            for (int i = 0; i < u.length; i++) {
//                dot += u[i] * v[i];
//                nru += u[i] * u[i];
//                nrv += v[i] * v[i];
//            }
//
//            double similarity = dot / (Math.sqrt(nru) * Math.sqrt(nrv));
//            return 1 - similarity;
//        }
//    }
//
//    static class DoubleInnerProduct implements DistanceFunction<double[], Double> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Double distance(double[] u, double[] v) {
//            double dot = 0;
//            for (int i = 0; i < u.length; i++) {
//                dot += u[i] * v[i];
//            }
//
//            return 1 - dot;
//        }
//    }
//
//    static class DoubleEuclideanDistance implements DistanceFunction<double[], Double> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Double distance(double[] u, double[] v) {
//            double sum = 0;
//            for (int i = 0; i < u.length; i++) {
//                double dp = u[i] - v[i];
//                sum += dp * dp;
//            }
//            return Math.sqrt(sum);
//        }
//    }
//
//
//    static class DoubleCanberraDistance implements DistanceFunction<double[], Double> {
//
//        private static final long serialVersionUID = 1L;
//
//        @Override
//        public Double distance(double[] u, double[] v) {
//            double sum = 0;
//            for (int i = 0; i < u.length; i++) {
//                double num = Math.abs(u[i] - v[i]);
//                double denom = Math.abs(u[i]) + Math.abs(v[i]);
//                sum += num == 0.0 && denom == 0.0 ? 0.0 : num / denom;
//            }
//            return sum;
//        }
//    }
//
//
//    static class DoubleBrayCurtisDistance implements DistanceFunction<double[], Double> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Double distance(double[] u, double[] v) {
//            double sump = 0;
//            double sumn = 0;
//
//            for (int i = 0; i < u.length; i++) {
//                sumn += Math.abs(u[i] - v[i]);
//                sump += Math.abs(u[i] + v[i]);
//            }
//
//            return sumn / sump;
//        }
//    }
//
//
//    static class DoubleCorrelationDistance implements DistanceFunction<double[], Double> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Double distance(double[] u, double[] v) {
//            double x = 0;
//            double y = 0;
//
//            for (int i = 0; i < u.length; i++) {
//                x += -u[i];
//                y += -v[i];
//            }
//
//            x /= u.length;
//            y /= v.length;
//
//            double num = 0;
//            double den1 = 0;
//            double den2 = 0;
//            for (int i = 0; i < u.length; i++) {
//                num += (u[i] + x) * (v[i] + y);
//
//                den1 += Math.abs(Math.pow(u[i] + x, 2));
//                den2 += Math.abs(Math.pow(v[i] + x, 2));
//            }
//
//            return 1 - (num / (Math.sqrt(den1) * Math.sqrt(den2)));
//        }
//    }
//
//    static class DoubleManhattanDistance implements DistanceFunction<double[], Double> {
//
//        private static final long serialVersionUID = 1L;
//
//
//        @Override
//        public Double distance(double[] u, double[] v) {
//            double sum = 0;
//            for (int i = 0; i < u.length; i++) {
//                sum += Math.abs(u[i] - v[i]);
//            }
//            return sum;
//        }
//    }
//
//}
