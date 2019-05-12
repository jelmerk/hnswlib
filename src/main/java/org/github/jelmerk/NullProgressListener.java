package org.github.jelmerk;

public class NullProgressListener implements ProgressListener {

    public static final NullProgressListener INSTANCE = new NullProgressListener();

    private NullProgressListener() {
    }

    @Override
    public void updateProgress(int workDone, int max) {

    }

}
