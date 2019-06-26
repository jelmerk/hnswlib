package com.github.jelmerk.knn.hnsw;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Bitset for tracking visited nodes.
 */
class BitSet implements Serializable {

    private static final long serialVersionUID = 1L;

    private int[] buffer;

    /**
     * Initializes a new instance of the {@link BitSet} class.
     *
     * @param count The number of items in the set.
     */
    BitSet(int count) {
        this.buffer = new int[(count >> 5) + 1];
    }

    /**
     * Checks whether the id is already in the set.
     *
     * @param id The identifier.
     * @return True if the identifier is in the set.
     */
    boolean contains(int id) {
        int carrier = this.buffer[id >> 5];
        return ((1 << (id & 31)) & carrier) != 0;
    }

    /**
     * Adds the id to the set.
     *
     * @param id The id to add
     */
    void add(int id)  {
        int mask = 1 << (id & 31);
        this.buffer[id >> 5] |= mask;
    }

    /**
     * Removes a id from the set.
     *
     * @param id The id to remove
     */
    void remove(int id) {
        int mask = 1 << (id & 31);
        this.buffer[id >> 5] &= ~mask;
    }

    /**
     * Clears the set.
     */
    void clear() {
        Arrays.fill(this.buffer, 0);
    }
}