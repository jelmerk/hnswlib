package com.github.jelmerk.knn.util;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Bitset.
 */
public class ArrayBitSet implements BitSet, Serializable {

    private static final long serialVersionUID = 1L;

    private final int[] buffer;

    /**
     * Initializes a new instance of the {@link ArrayBitSet} class.
     *
     * @param count The number of items in the set.
     */
    public ArrayBitSet(int count) {
        this.buffer = new int[(count >> 5) + 1];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean contains(int id) {
        int carrier = this.buffer[id >> 5];
        return ((1 << (id & 31)) & carrier) != 0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(int id)  {
        int mask = 1 << (id & 31);
        this.buffer[id >> 5] |= mask;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void remove(int id) {
        int mask = 1 << (id & 31);
        this.buffer[id >> 5] &= ~mask;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void clear() {
        Arrays.fill(this.buffer, 0);
    }
}