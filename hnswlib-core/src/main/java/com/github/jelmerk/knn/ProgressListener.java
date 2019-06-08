package com.github.jelmerk.knn;

/**
 * Callback interface for reporting on the progress of an index operation.
 */
@FunctionalInterface
public interface ProgressListener {

    /**
     * Called by the index at set intervals to report progress of the indexing process.
     *
     * @param workDone the amount of items indexed so far
     * @param max the total amount of items to be indexed
     */
    void updateProgress(int workDone, int max);
}
