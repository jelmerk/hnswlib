package com.github.jelmerk.hnswlib.core.hnsw;

import com.github.jelmerk.hnswlib.core.IndexException;

/**
 * Thrown to indicate the size of the index has been exceeded.
 */
public class SizeLimitExceededException extends IndexException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a SizeLimitExceededException with the specified detail message.
     *
     * @param message the detail message.
     */
    public SizeLimitExceededException(String message) {
        super(message);
    }
}
