package com.github.jelmerk.knn.statistics;

import java.io.Serializable;
import java.util.Objects;

/**
 * Holds statistics about the index.
 */
public class IndexStats implements Serializable {

    private static final long serialVersionUID = 1L;

    private final double precision;

    /**
     * Constructs a new IndexStats instance.
     *
     * @param precision the precision as a value between 0 and 1
     */
    public IndexStats(double precision) {
        this.precision = precision;
    }

    /**
     * Precision of the index as a value between 0 and 1 where 1 means being correct 100% of the time and 0 means always
     * being wrong.
     *
     * @return the precision as a value between 0 and 1
     */
    public double precision() {
        return precision;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IndexStats that = (IndexStats) o;
        return Double.compare(that.precision, precision) == 0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        return Objects.hash(precision);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return "IndexStats{" +
                "precision=" + precision +
                '}';
    }
}
