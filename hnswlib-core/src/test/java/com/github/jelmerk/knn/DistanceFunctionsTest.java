package com.github.jelmerk.knn;

import org.junit.jupiter.api.Test;

import static org.hamcrest.Matchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

class DistanceFunctionsTest {

    private SparseVector<float[]> sparseFloatVector1 = new SparseVector<>(new int[] {0, 1, 2}, new float[] { 0.01f, 0.02f, 0.03f } );
    private SparseVector<float[]> sparseFloatVector2 = new SparseVector<>(new int[] {0, 1, 2}, new float[] { 0.03f, 0.02f, 0.01f } );

    private SparseVector<double[]> sparseDoubleVector1 = new SparseVector<>(new int[] {0, 1, 2}, new double[] { 0.01d, 0.02d, 0.03d } );
    private SparseVector<double[]> sparseDoubleVector2 = new SparseVector<>(new int[] {0, 1, 2}, new double[] { 0.03d, 0.02d, 0.01d } );

    private float[] floatVector1 = new float[] { 0.01f, 0.02f, 0.03f };
    private float[] floatVector2 = new float[] { 0.03f, 0.02f, 0.01f };

    private double[] doubleVector1 = new double[] { 0.01d, 0.02d, 0.03d };
    private double[] doubleVector2 = new double[] { 0.03d, 0.02d, 0.01d };

    private double error = 1e-4;

    @Test
    void floatSparseVectorInnerProduct() {
        assertThat((double)DistanceFunctions.FLOAT_SPARSE_VECTOR_INNER_PRODUCT.distance(sparseFloatVector1, sparseFloatVector2), closeTo(0.999, error));
    }

    @Test
    void floatCosineDistance() {
        assertThat((double) DistanceFunctions.FLOAT_COSINE_DISTANCE.distance(floatVector1, floatVector2), closeTo(0.28571433, error));
    }

    @Test
    void floatInnerProduct() {
        assertThat((double) DistanceFunctions.FLOAT_INNER_PRODUCT.distance(floatVector1, floatVector2), closeTo(0.999, error));
    }

    @Test
    void floatEuclideanDistance() {
        assertThat((double) DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE.distance(floatVector1, floatVector2), closeTo(0.02828427, error));
    }

    @Test
    void floatCanberraDistance() {
        assertThat((double) DistanceFunctions.FLOAT_CANBERRA_DISTANCE.distance(floatVector1, floatVector2), closeTo(1, error));
    }

    @Test
    void floatBrayCurtisDistance() {
        assertThat((double) DistanceFunctions.FLOAT_BRAY_CURTIS_DISTANCE.distance(floatVector1, floatVector2), closeTo(0.3333333, error));
    }

    @Test
    void floatCorrelationDistance() {
        assertThat((double) DistanceFunctions.FLOAT_CORRELATION_DISTANCE.distance(floatVector1, floatVector2), closeTo(2, error));
    }

    @Test
    void floatManhattanDistance() {
        assertThat((double) DistanceFunctions.FLOAT_MANHATTAN_DISTANCE.distance(floatVector1, floatVector2), closeTo(0.04, error));
    }

    @Test
    void doubleSparseVectorInnerProduct() {
        assertThat(DistanceFunctions.DOUBLE_SPARSE_VECTOR_INNER_PRODUCT.distance(sparseDoubleVector1, sparseDoubleVector2), closeTo(0.999, error));
    }

    @Test
    void doubleCosineDistance() {
        assertThat(DistanceFunctions.DOUBLE_COSINE_DISTANCE.distance(doubleVector1, doubleVector2), closeTo(0.2857142857142858, error));
    }

    @Test
    void doubleInnerProduct() {
        assertThat(DistanceFunctions.DOUBLE_INNER_PRODUCT.distance(doubleVector1, doubleVector2), closeTo(0.999, error));
    }

    @Test
    void doubleEuclideanDistance() {
        assertThat(DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE.distance(doubleVector1, doubleVector2), closeTo(0.028284271247461898, error));
    }

    @Test
    void doubleCanberraDistance() {
        assertThat(DistanceFunctions.DOUBLE_CANBERRA_DISTANCE.distance(doubleVector1, doubleVector2), closeTo(1, error));
    }

    @Test
    void doubleBrayCurtisDistance() {
        assertThat(DistanceFunctions.DOUBLE_BRAY_CURTIS_DISTANCE.distance(doubleVector1, doubleVector2), closeTo(0.3333333333333333, error));
    }

    @Test
    void doubleCorrelationDistance() {
        assertThat(DistanceFunctions.DOUBLE_CORRELATION_DISTANCE.distance(doubleVector1, doubleVector2), closeTo(2, error));
    }

    @Test
    void doubleManhattanDistance() {
        assertThat(DistanceFunctions.DOUBLE_MANHATTAN_DISTANCE.distance(doubleVector1, doubleVector2), closeTo(0.04, error));
    }
}
