package dev.m00nl1ght.clockwork.profiler.impl;

import dev.m00nl1ght.clockwork.profiler.ProfilerEntry;
import dev.m00nl1ght.clockwork.profiler.ProfilerGroup;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class SimpleProfilerGroup extends ProfilerGroup {

    protected final Map<String, ProfilerEntry> entries = new LinkedHashMap<>();
    protected final Map<String, ProfilerGroup> groups = new LinkedHashMap<>();

    public SimpleProfilerGroup(String name) {
        super(name);
    }

    public void addEntry(ProfilerEntry entry) {
        final var existing = entries.putIfAbsent(entry.getName(), entry);
        if (existing != null) throw new IllegalArgumentException("entry name duplicate");
    }

    public void addEntries(ProfilerEntry... entries) {
        for (var entry : entries) addEntry(entry);
    }

    public void addGroup(ProfilerGroup group) {
        final var existing = groups.putIfAbsent(group.getName(), group);
        if (existing != null) throw new IllegalArgumentException("group name duplicate");
    }

    public void addGroups(ProfilerGroup... groups) {
        for (var group : groups) addGroup(group);
    }

    @Override
    public List<ProfilerEntry> getEntries() {
        return List.copyOf(entries.values());
    }

    @Override
    public List<ProfilerGroup> getGroups() {
        return List.copyOf(groups.values());
    }

    public SimpleCyclicProfilerEntry entry(String name) {
        return (SimpleCyclicProfilerEntry) entries.computeIfAbsent(name, n -> new SimpleCyclicProfilerEntry(n, 100));
    }

    @SuppressWarnings("unchecked")
    public <T extends ProfilerEntry> T entry(String name, Function<String, T> factory) {
        return (T) entries.computeIfAbsent(name, factory);
    }

    public SimpleProfilerGroup group(String name) {
        return (SimpleProfilerGroup) groups.computeIfAbsent(name, SimpleProfilerGroup::new);
    }

    @SuppressWarnings("unchecked")
    public <T extends ProfilerGroup> T group(String name, Function<String, T> factory) {
        return (T) groups.computeIfAbsent(name, factory);
    }

}
