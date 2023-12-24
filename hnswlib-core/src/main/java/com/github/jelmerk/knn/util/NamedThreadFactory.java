package com.github.jelmerk.knn.util;

import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A {@link ThreadFactory} implementation that names each thread created using a user defined format string.
 */
public class NamedThreadFactory implements ThreadFactory {

    private final String namingPattern;
    private final AtomicInteger counter;

    /**
     * Constructs a new NamedThreadFactory
     *
     * @param namingPattern format string used to construct the name assigned to each thread created.
     */
    public NamedThreadFactory(String namingPattern) {
        this.namingPattern = namingPattern;
        this.counter = new AtomicInteger(0);
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public Thread newThread(Runnable r) {
        return new Thread(r, String.format(namingPattern, counter.incrementAndGet()));
    }
}
