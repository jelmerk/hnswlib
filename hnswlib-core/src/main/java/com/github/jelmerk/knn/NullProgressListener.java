package com.github.jelmerk.knn;

/**
 * Implementation of {@link ProgressListener} that does nothing.
 */
public class NullProgressListener implements ProgressListener {

    /**
     * Singleton instance of {@link NullProgressListener}.
     */
    public static final NullProgressListener INSTANCE = new NullProgressListener();

    private NullProgressListener() {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void updateProgress(int workDone, int max) {
        // do nothing
    }

}
