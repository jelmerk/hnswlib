package org.github.jelmerk.knn;

/**
 * Calculates cosine similarity.
 *
 * Intuition behind selecting float as a carrier.
 *
 * 1. In practice we work with vectors of dimensionality 100 and each component has value in range [-1; 1]
 *    There certainly is a possibility of underflow.
 *    But we assume that such cases are rare and we can rely on such underflow losses.
 *
 * 2. According to the article http://www.ti3.tuhh.de/paper/rump/JeaRu13.pdf
 *    the floating point rounding error is less then 100 * 2^-24 * sqrt(100) * sqrt(100) &lt; 0.0005960
 *    We deem such precision is satisfactory for out needs.
 */
public class DistanceFunctions {

    // run with  -XX:+UseSuperWord -XX:+UnlockDiagnosticVMOptions -XX:CompileCommand=print,*CosineDistance.cosineDistance to see if simd is being used
    // see https://cr.openjdk.java.net/~vlivanov/talks/2017_Vectorization_in_HotSpot_JVM.pdf

    /**
     * Calculates cosine distance.
     *
     * @param u Left vector.
     * @param v Right vector.
     * @return Cosine distance between u and v.
     */
    public static float cosineDistance(float[] u, float[] v)  {

        if (u.length != v.length) {
            throw new IllegalArgumentException("Vectors have non-matching dimensions");
        }

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
        if (u.length != v.length) {
            throw new IllegalArgumentException("Vectors have non-matching dimensions");
        }

        float dot = 0;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
        }

        return 1 - dot;
    }

}
