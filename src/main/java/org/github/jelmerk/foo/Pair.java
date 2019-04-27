package org.github.jelmerk.foo;

import java.util.Objects;

public class Pair<A, B> {

    private final A first;
    private final B b;

    public Pair(A first, B second) {
        this.first = first;
        this.b = second;
    }

    public A getFirst() {
        return first;
    }

    public B getSecond() {
        return b;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Pair<?, ?> pair = (Pair<?, ?>) o;
        return Objects.equals(first, pair.first) &&
                Objects.equals(b, pair.b);
    }

    @Override
    public int hashCode() {
        return Objects.hash(first, b);
    }
}
