package org.github.jelmerk.knn;

/**
 * Implementation of {@link ProgressListener} that does nothing.
 */
class NullProgressListener implements ProgressListener {

    /**
     * Singleton instance of {@link NullProgressListener}.
     */
    static final NullProgressListener INSTANCE = new NullProgressListener();

    private NullProgressListener() {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void updateProgress(int workDone, int max) {

    }

}
