package org.github.jelmerk.hnsw;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Bitset for tracking visited nodes.
 */
class VisitedBitSet implements Serializable {

    private int[] buffer;

    /**
     * Initializes a new instance of the {@link VisitedBitSet} class.
     *
     * @param nodesCount The number of nodes to track in the set.
     */
    VisitedBitSet(int nodesCount) {
        this.buffer = new int[(nodesCount >> 5) + 1];
    }

    /**
     * Checks whether the node is already in the set.
     *
     * @param nodeId The identifier of the node.
     * @return True if the node is in the set.
     */
    boolean contains(int nodeId) {
        int carrier = this.buffer[nodeId >> 5];
        return ((1 << (nodeId & 31)) & carrier) != 0;
    }

    /**
     * Adds the node id to the set.
     *
     * @param nodeId The node id to add
     */
    void add(int nodeId)  {
        int mask = 1 << (nodeId & 31);
        this.buffer[nodeId >> 5] |= mask;
    }

    /**
     * Clears the set.
     */
    void clear() {
        Arrays.fill(this.buffer, 0);
    }
}