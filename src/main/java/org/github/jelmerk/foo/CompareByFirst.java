package org.github.jelmerk.foo;

import java.util.Comparator;

public class CompareByFirst<A extends Comparable<A>, B> implements Comparator<Pair<A, B>> {
    @Override
    public int compare(Pair<A, B> o1, Pair<A, B> o2) {
        return o1.getFirst().compareTo(o2.getFirst());
    }
}

