package com.github.jelmerk.knn;

import org.junit.jupiter.api.Test;

import static org.hamcrest.Matchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

class DistanceFunctionsTest {

    private float[] floatVector1 = new float[] { 0.01f, 0.02f, 0.03f };
    private float[] floatVector2 = new float[] { 0.03f, 0.02f, 0.01f };

    private double[] doubleVector1 = new double[] { 0.01d, 0.02d, 0.03d };
    private double[] doubleVector2 = new double[] { 0.03d, 0.02d, 0.01d };

    @Test
    void floatCosineDistance() {
        assertThat(DistanceFunctions.FLOAT_COSINE_DISTANCE.distance(floatVector1, floatVector2), is(0.28571433f));
    }

    @Test
    void floatInnerProduct() {
        assertThat(DistanceFunctions.FLOAT_INNER_PRODUCT.distance(floatVector1, floatVector2), is(0.999f));
    }

    @Test
    void floatEuclideanDistance() {
        assertThat(DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2), is(0.9724937f));
    }

    @Test
    void floatCanberraDistance() {
        assertThat(DistanceFunctions.FLOAT_CANBERRA_DISTANCE.distance(floatVector1, floatVector2), is(1f));
    }

    @Test
    void floatBrayCurtisDistance() {
        assertThat(DistanceFunctions.FLOAT_BRAY_CURTIS_DISTANCE.distance(floatVector1, floatVector2), is(0.33333334f));
    }

    @Test
    void floatCorrelationDistance() {
        assertThat(DistanceFunctions.FLOAT_CORRELATION_DISTANCE.distance(floatVector1, floatVector2), is(2f));
    }

    @Test
    void floatManhattanDistance() {
        assertThat(DistanceFunctions.FLOAT_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2), is(0.04F));
    }

    @Test
    void doubleCosineDistance() {
        assertThat(DistanceFunctions.DOUBLE_COSINE_DISTANCE.distance(doubleVector1, doubleVector2), is(0.2857142857142858d));
    }

    @Test
    void doubleInnerProduct() {
        assertThat(DistanceFunctions.DOUBLE_INNER_PRODUCT.distance(doubleVector1, doubleVector2), is(0.999d));
    }

    @Test
    void doubleEuclideanDistance() {
        assertThat(DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE.distance(doubleVector1, doubleVector2), is(0.9724937237315234d));
    }

    @Test
    void doubleCanberraDistance() {
        assertThat(DistanceFunctions.DOUBLE_CANBERRA_DISTANCE.distance(doubleVector1, doubleVector2), is(0.9999999999999998d));
    }

    @Test
    void doubleBrayCurtisDistance() {
        assertThat(DistanceFunctions.DOUBLE_BRAY_CURTIS_DISTANCE.distance(doubleVector1, doubleVector2), is(0.3333333333333333d));
    }

    @Test
    void doubleCorrelationDistance() {
        assertThat(DistanceFunctions.DOUBLE_CORRELATION_DISTANCE.distance(doubleVector1, doubleVector2), is(2d));
    }

    @Test
    void doubleManhattanDistance() {
        assertThat(DistanceFunctions.DOUBLE_MANHATTAN_DISTANCE.distance(doubleVector1, doubleVector2), is(0.039999999999999994d));
    }
}
