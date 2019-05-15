package org.github.jelmerk.knn.hnsw;

/**
 * Auxiliary tools for working with distance variables.
 */
final class DistanceUtils {

    private DistanceUtils() {
    }

    /**
     * Distance is Lower Than.
     *
     * @param x Left argument.
     * @param y Right argument.
     * @param <TDistance> The type of the distance.
     * @return True if x &lt; y.
     */
    static <TDistance extends Comparable<TDistance>> boolean lt(TDistance x, TDistance y) {
        return x.compareTo(y) < 0;
    }

    /**
     * Distance is Greater Than.
     *
     * @param x Left argument.
     * @param y Right argument.
     * @param <TDistance> The type of the distance.
     * @return True if x &gt; y.
     */
    static <TDistance extends Comparable<TDistance>> boolean gt(TDistance x, TDistance y) {
        return x.compareTo(y) > 0;
    }

}
