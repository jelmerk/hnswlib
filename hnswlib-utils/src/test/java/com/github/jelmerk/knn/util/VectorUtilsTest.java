package com.github.jelmerk.knn.util;

import static org.hamcrest.CoreMatchers.*;
import static org.junit.Assert.*;
import org.junit.Test;

import static com.github.jelmerk.knn.util.VectorUtils.*;

public class VectorUtilsTest {

    private float[] floatVector = { 0.01f, 0.02f, 0.03f };

    private double[] doubleVector = { 0.01d, 0.02d, 0.03d };

    @Test
    public void calculateFloatVectorMagnitude() {
        assertThat(magnitude(floatVector), is(0.037416574f));
    }

    @Test
    public void calculateDoubleVectorMagnitude() {
        assertThat(magnitude(doubleVector), is(0.03741657386773942d));
    }

    @Test
    public void normalizeFloatVector() {
        assertThat(normalize(floatVector), is(new float[] { 0.26726124f, 0.5345225f, 0.8017837f }));
    }

    @Test
    public void normalizeDoubleVector() {
        assertThat(normalize(doubleVector), is(new double[] { 0.2672612419124244d, 0.5345224838248488d, 0.8017837257372731d }));
    }
}
