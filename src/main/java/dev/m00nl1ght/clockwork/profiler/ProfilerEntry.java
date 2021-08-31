package dev.m00nl1ght.clockwork.profiler;

public abstract class ProfilerEntry {

    private final String name;
    private long lastStart = -1;

    protected ProfilerEntry(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public abstract void put(int value);

    public void put(long value) {
        if (value > Integer.MAX_VALUE) value = Integer.MAX_VALUE;
        put((int) value);
    }

    public abstract int get(int idx);

    public abstract int getSize();

    public abstract int getCount();

    public abstract int getAverage();

    public abstract int getMax();

    public abstract int getMin();

    public void begin() {
        if (lastStart != -1) throw new IllegalStateException();
        lastStart = System.nanoTime();
    }

    public void end() {
        if (lastStart == -1) throw new IllegalStateException();
        put(System.nanoTime() - lastStart);
        lastStart = -1;
    }

    @Override
    public String toString() {
        return name + " (" + getCount() + ") => AVG ~ " + getAverage() + " MIN " + getMin() + " MAX " + getMax();
    }

}
