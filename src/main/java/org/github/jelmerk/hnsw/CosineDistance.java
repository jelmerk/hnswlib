package org.github.jelmerk.hnsw;

public class CosineDistance {


    // TODO boxed primitives are not great for performance nor memory usage
    public static Float nonOptimized(Float[] u, Float[] v)  {
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


    // TODO boxed primitives are not great for performance nor memory usage
    public static Float forUnits(Float[] u, Float[] v) {
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
