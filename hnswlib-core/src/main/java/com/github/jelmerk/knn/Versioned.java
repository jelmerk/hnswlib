package com.github.jelmerk.knn;

/**
 * Items implementing this interface are versioned. This allows the index to make sure that the correct items are kept
 * when items are added to the index out of order. by rejecting items with a version number lower than the one stored
 * in the index already.
 *
 * @param <TVersion> the version
 */
public interface Versioned<TVersion extends Comparable<TVersion>> {

    /**
     * The version of the item.
     *
     * @return version of the item
     */
    TVersion version();

}
