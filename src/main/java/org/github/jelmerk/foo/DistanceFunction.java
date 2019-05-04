package org.github.jelmerk.foo;

@FunctionalInterface
public interface DistanceFunction<T> {

    float distance(float[] u, float[] v, T param);

}