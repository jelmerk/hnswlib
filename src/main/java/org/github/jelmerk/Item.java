package org.github.jelmerk;

import java.io.Serializable;

public interface Item<TId, TVector> extends Serializable {

    TId getId();

    TVector getVector();
}
