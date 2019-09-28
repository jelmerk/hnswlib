package com.github.jelmerk.knn.util;

import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

public class NamedThreadFactory implements ThreadFactory {

    private final String namingPattern;
    private final AtomicInteger counter;

    public NamedThreadFactory(String namingPattern) {
        this.namingPattern = namingPattern;
        this.counter = new AtomicInteger(0);
    }

    @Override
    public Thread newThread(Runnable r) {
        return new Thread(r, String.format(namingPattern, counter.incrementAndGet()));
    }
}
