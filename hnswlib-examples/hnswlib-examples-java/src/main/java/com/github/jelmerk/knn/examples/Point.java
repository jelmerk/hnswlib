package com.github.jelmerk.knn.examples;

import com.github.jelmerk.knn.Item;

import java.util.Arrays;

public class Point implements Item<Integer, double[]> {

    private static final long serialVersionUID = 1L;

    private final Integer id;
    private final double[] vector;

    Point(Integer id, double[] vector) {
        this.id = id;
        this.vector = vector;
    }

    @Override
    public Integer id() {
        return id;
    }

    @Override
    public double[] vector() {
        return vector;
    }

    @Override
    public String toString() {
        return "Point{" +
                "id='" + id + '\'' +
                ", vector=" + Arrays.toString(vector) +
                '}';
    }
    public String printPoint() {
        return "("+vector[0]+","+vector[1]+")";

    }



}
