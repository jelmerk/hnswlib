package org.github.jelmerk.foo;

public interface DistanceFunction<Item, TDistance extends Comparable<TDistance>> {

    TDistance distance(Item u, Item v);

}