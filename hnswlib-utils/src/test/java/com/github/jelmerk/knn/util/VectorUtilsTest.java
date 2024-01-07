package com.github.jelmerk.knn.util;

import org.junit.jupiter.api.Test;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;
import static com.github.jelmerk.knn.util.VectorUtils.*;

class VectorUtilsTest {

    private final float[] floatVector = { 0.01f, 0.02f, 0.03f };

    private final double[] doubleVector = { 0.01d, 0.02d, 0.03d };

    @Test
    void calculateFloatVectorMagnitude() {
        assertThat(magnitude(floatVector), is(0.037416574f));
    }

    @Test
    void calculateDoubleVectorMagnitude() {
        assertThat(magnitude(doubleVector), is(0.03741657386773942d));
    }

    @Test
    void normalizeFloatVector() {
        assertThat(normalize(floatVector), is(new float[] { 0.26726124f, 0.5345225f, 0.8017837f }));
    }

    @Test
    void normalizeDoubleVector() {
        assertThat(normalize(doubleVector), is(new double[] { 0.2672612419124244d, 0.5345224838248488d, 0.8017837257372731d }));
    }
}
