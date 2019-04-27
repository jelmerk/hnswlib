package org.github.jelmerk.hnsw;

/**
 * Type of heuristic to select best neighbours for a node.
 */
public enum NeighbourSelectionHeuristic {
    /**
     * Marker for the Algorithm 3 (SELECT-NEIGHBORS-SIMPLE) from the article.
     * Implemented in {@link org.github.jelmerk.hnsw.Node.Algorithm3}
     */
    SELECT_SIMPLE,

    /**
     * Marker for the Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) from the article.
     * Implemented in {@link org.github.jelmerk.hnsw.Node.Algorithm4}
     */
    SELECT_HEURISTIC
}