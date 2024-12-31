package com.github.jelmerk.knn.util;

import java.io.Serializable;
import java.util.Comparator;

/**
 * Implementation of {@link Comparator} that is serializable and throws {@link UnsupportedOperationException} when
 * compare is called. Useful as a dummy placeholder when you know it will never be called.
 *
 * @param <T> the type of objects that may be compared by this comparator
 */
public class DummyComparator<T> implements Comparator<T>, Serializable {

    @Override
    public int compare(T o1, T o2) {
        throw new UnsupportedOperationException();
    }
}
