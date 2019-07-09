package com.github.jelmerk.knn.util;

public interface BitSet {

    /**
     * Checks whether the id is already in the set.
     *
     * @param id The identifier.
     * @return True if the identifier is in the set.
     */
    boolean contains(int id);

    /**
     * Adds the id to the set.
     *
     * @param id The id to add
     */
    void add(int id);

    /**
     * Removes a id from the set.
     *
     * @param id The id to remove
     */
    void remove(int id);

    /**
     * Clears the set.
     */
    void clear();
}