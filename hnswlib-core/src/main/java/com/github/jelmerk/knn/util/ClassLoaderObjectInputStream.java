package com.github.jelmerk.knn.util;


import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;
import java.io.StreamCorruptedException;
import java.lang.reflect.Proxy;

/**
 * A special ObjectInputStream that loads a class based on a specified ClassLoader rather than the system default.
 */
public class ClassLoaderObjectInputStream extends ObjectInputStream {

    /** The class loader to use. */
    private final ClassLoader classLoader;

    /**
     * Constructs a new ClassLoaderObjectInputStream.
     *
     * @param classLoader  the ClassLoader from which classes should be loaded
     * @param inputStream  the InputStream to work on
     * @throws IOException in case of an I/O error
     * @throws StreamCorruptedException if the stream is corrupted
     */
    public ClassLoaderObjectInputStream(ClassLoader classLoader, InputStream inputStream) throws IOException, StreamCorruptedException {
        super(inputStream);
        this.classLoader = classLoader;
    }

    /**
     * Resolve a class specified by the descriptor using the
     * specified ClassLoader or the super ClassLoader.
     *
     * @param objectStreamClass  descriptor of the class
     * @return the Class object described by the ObjectStreamClass
     * @throws IOException in case of an I/O error
     * @throws ClassNotFoundException if the Class cannot be found
     */
    @Override
    protected Class<?> resolveClass(ObjectStreamClass objectStreamClass) throws IOException, ClassNotFoundException {

        Class<?> clazz = Class.forName(objectStreamClass.getName(), false, classLoader);

        if (clazz != null) {
            return clazz;
        } else {
            return super.resolveClass(objectStreamClass);
        }
    }

    /**
     * Create a proxy class that implements the specified interfaces using the specified ClassLoader or the super ClassLoader.
     *
     * @param interfaces the interfaces to implement
     * @return a proxy class implementing the interfaces
     * @throws IOException in case of an I/O error
     * @throws ClassNotFoundException if the Class cannot be found
     * @see java.io.ObjectInputStream#resolveProxyClass(java.lang.String[])
     */
    @Override
    protected Class<?> resolveProxyClass(String[] interfaces) throws IOException, ClassNotFoundException {
        Class<?>[] interfaceClasses = new Class[interfaces.length];
        for (int i = 0; i < interfaces.length; i++) {
            interfaceClasses[i] = Class.forName(interfaces[i], false, classLoader);
        }
        try {
            return Proxy.getProxyClass(classLoader, interfaceClasses);
        } catch (IllegalArgumentException e) {
            return super.resolveProxyClass(interfaces);
        }
    }

}

