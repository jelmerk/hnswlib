package com.github.jelmerk.knn;

/**
 * Base class for exceptions thrown by {@link Index} implementations.
 */
public abstract class IndexException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs an IndexException
     */
    public IndexException() {
    }

    /**
     * Constructs a IndexException with the specified detail message.
     *
     * @param message the detail message.
     */
    public IndexException(String message) {
        super(message);
    }

    /**
     * Constructs a IndexException with the specified detail message and cause.
     *
     * @param message the detail message.
     * @param  cause the cause
     */
    public IndexException(String message, Throwable cause) {
        super(message, cause);
    }


}
