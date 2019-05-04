package org.github.jelmerk.foo;

public interface SpaceInterface<T> {

    int getDataSize();

    DistanceFunction<T> getDistanceFunction();

    T getDistanceFunctionParam();

}