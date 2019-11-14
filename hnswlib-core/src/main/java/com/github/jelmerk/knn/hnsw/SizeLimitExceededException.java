package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.IndexException;

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
