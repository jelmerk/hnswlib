package com.github.jelmerk.knn.util;

import org.junit.jupiter.api.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;

public class ArrayBitSetTest {

    @Test
    void copyConstructor() {
        ArrayBitSet bitset = new ArrayBitSet(100);
        bitset.add(50);
        ArrayBitSet other = new ArrayBitSet(bitset, 200);
        other.add(101);
        assertThat(other.contains(50), is(true));
        assertThat(other.contains(101), is(true));
    }
}
