package com.github.jelmerk.knn.statistics;

import java.io.Serializable;
import java.util.Objects;

public class IndexStats implements Serializable {

    private static final long serialVersionUID = 1L;

    private final double precision;

    public IndexStats(double precision) {
        this.precision = precision;
    }

    public double precision() {
        return precision;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IndexStats that = (IndexStats) o;
        return Double.compare(that.precision, precision) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(precision);
    }

    @Override
    public String toString() {
        return "IndexStats{" +
                "precision=" + precision +
                '}';
    }
}
