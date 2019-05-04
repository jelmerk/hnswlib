package org.github.jelmerk.foo;

@FunctionalInterface
public interface DistanceFunction {

    float distance(float[] u, float[] v);

}