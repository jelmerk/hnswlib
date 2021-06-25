package com.github.jelmerk.knn;

/**
 * Thrown to indicate that a nested exception occurred in one of the worker threads.
 */
public class IndexNestedException extends IndexException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a IndexNestedException with the specified detail message and cause.
     *
     * @param message the detail message.
     * @param  cause the cause.
     */
    public IndexNestedException(String message, Throwable cause) {
        super(message, cause);
    }
}
