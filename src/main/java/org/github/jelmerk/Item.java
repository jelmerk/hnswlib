package org.github.jelmerk;

public interface Item<TId, TVector> {

    TId getId();

    TVector getVector();
}
