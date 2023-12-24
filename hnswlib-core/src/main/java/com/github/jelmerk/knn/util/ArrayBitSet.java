package com.github.jelmerk.knn.util;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Bitset.
 */
public class ArrayBitSet implements Serializable {

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
     * Initializes a new instance of the {@link ArrayBitSet} class. and copies the values of another bitset.
     *
     * @param other other bitset
     * @param count The number of items in the set.
     */
    public ArrayBitSet(ArrayBitSet other, int count) {
        this.buffer = Arrays.copyOf(other.buffer, (count >> 5) + 1);
    }

    /**
     * Returns true if this set contains the specified element
     */
    public boolean contains(int bitIndex) {
        int carrier = this.buffer[bitIndex >> 5];
        return ((1 << (bitIndex & 31)) & carrier) != 0;
    }

    /**
     * Add element to the bitset.
     *
     * @param id element to add
     */
    public void add(int id)  {
        int mask = 1 << (id & 31);
        this.buffer[id >> 5] |= mask;
    }

    /**
     * Removes element from the bitset.
     *
     * @param id element to remove
     */
    public void remove(int id) {
        int mask = 1 << (id & 31);
        this.buffer[id >> 5] &= ~mask;
    }

    /**
     * Clears the bitset.
     */
    public void clear() {
        Arrays.fill(this.buffer, 0);
    }
}