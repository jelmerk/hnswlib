package com.github.jelmerk.knn.util;

import java.io.Serializable;

public class SynchronizedBitSet implements BitSet, Serializable  {

    private static final long serialVersionUID = 1L;

    private final BitSet delegate;

    /**
     * Constructs a new SynchronizedBitSet.
     *
     * @param delegate the wrapped bitset
     */
    public SynchronizedBitSet(BitSet delegate) {
        this.delegate = delegate;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public synchronized boolean contains(int id) {
        return delegate.contains(id);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public synchronized void add(int id) {
        delegate.add(id);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public synchronized void remove(int id) {
        delegate.remove(id);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public synchronized void clear() {
        delegate.clear();
    }
}