package dev.m00nl1ght.nnLoom.profiler.impl;

import dev.m00nl1ght.nnLoom.profiler.ProfilerEntry;

public class SimpleCyclicProfilerEntry extends ProfilerEntry {

    private final int[] data;
    private int count = 0;
    private int min = Integer.MAX_VALUE;
    private int max = Integer.MIN_VALUE;
    private long total = 0;

    public SimpleCyclicProfilerEntry(String name) {
        this(name, 100);
    }

    public SimpleCyclicProfilerEntry(String name, int capacity) {
        super(name);
        this.data = new int[capacity];
    }

    @Override
    public void put(int value) {
        final var pointer = count % data.length;
        if (value < min) min = value;
        if (value > max) max = value;
        total += value;
        total -= data[pointer];
        data[pointer] = value;
        count++;
    }

    @Override
    public int get(int idx) {
        if (idx < 0 || idx > data.length || idx > count) throw new ArrayIndexOutOfBoundsException();
        return data[count <= data.length ? idx : (count + idx) % data.length];
    }

    @Override
    public int getSize() {
        return Math.min(data.length, count);
    }

    @Override
    public int getCount() {
        return count;
    }

    @Override
    public int getAverage() {
        return count == 0 ? 0 : (int) total / getSize();
    }

    @Override
    public int getMax() {
        return count == 0 ? 0 : max;
    }

    @Override
    public int getMin() {
        return count == 0 ? 0 : min;
    }

}
