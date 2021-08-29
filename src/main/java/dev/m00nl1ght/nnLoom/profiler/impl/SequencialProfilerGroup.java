package dev.m00nl1ght.nnLoom.profiler.impl;

import dev.m00nl1ght.nnLoom.profiler.ProfilerEntry;
import dev.m00nl1ght.nnLoom.profiler.ProfilerGroup;

import java.util.Collection;
import java.util.List;
import java.util.Objects;

public class SequencialProfilerGroup extends ProfilerGroup {

    protected final ProfilerEntry[] entries;

    private int idx = -1;
    private long lastBegin = -1L;

    public SequencialProfilerGroup(String name, Collection<? extends ProfilerEntry> entries) {
        super(name);
        this.entries = Objects.requireNonNull(entries).toArray(ProfilerEntry[]::new);
    }

    @Override
    public List<ProfilerEntry> getEntries() {
        return List.of(entries);
    }

    public void beginSequence() {
        if (idx > 0) throw new IllegalStateException();
        idx = 0; lastBegin = -1L;
    }

    public void endSequence() {
        if (idx != entries.length) throw new IllegalStateException();
        idx = -1; lastBegin = -1L;
    }

    public void begin() {
        if (idx < 0 || idx >= entries.length || lastBegin >= 0L) throw new IllegalStateException();
        lastBegin = System.nanoTime();
    }

    public void end() {
        if (lastBegin < 0L) throw new IllegalStateException();
        entries[idx].put(System.nanoTime() - lastBegin);
        idx++;
    }

    public void step(Runnable runnable) {
        begin();
        runnable.run();
        end();
    }

}
