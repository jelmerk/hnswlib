package com.github.jelmerk.knn;

import java.io.Serializable;

/**
 * A sparse vector represented by an index array and a value array.
 */
public class SparseVector<TVector> implements Serializable {

    private static final long serialVersionUID = 1L;

    private int[] indices;
    private TVector values;

    /**
     * Constructs a new SparseVector instance.
     *
     * @param indices the index array, must be in ascending order
     * @param values the values array
     */
    public SparseVector(int[] indices, TVector values) {
        this.indices = indices;
        this.values = values;
    }
    /**
     * Returns the index array. Values are returned in ascending order.
     *
     * @return the index array
     */
    public int[] indices() {
        return indices;
    }

    /**
     * Returns the values array.
     *
     * @return the values array.
     */
    public TVector values() {
        return values;
    }
}
