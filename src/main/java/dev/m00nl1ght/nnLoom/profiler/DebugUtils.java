package dev.m00nl1ght.nnLoom.profiler;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.stream.Collectors;

public class DebugUtils {

    private DebugUtils() {}

    public static String printProfilerInfo(DebugProfiler profiler) {
        final var builder = new StringBuilder();
        builder.append("########## Debug Profiler ##########\n");
        for (var g : profiler.getGroups()) printProfilerInfo(builder, g, 0);
        builder.append("####################################");
        return builder.toString();
    }

    private static void printProfilerInfo(StringBuilder builder, ProfilerGroup group, int ind) {
        builder.append("  ".repeat(ind)).append('[').append(group.getName()).append(']').append('\n');
        for (var e : group.getEntries()) printProfilerInfo(builder, e, ind + 1);
        for (var g : group.getGroups()) printProfilerInfo(builder, g, ind + 1);
    }

    private static void printProfilerInfo(StringBuilder builder, ProfilerEntry entry, int ind) {
        builder.append("  ".repeat(ind)).append(entry.toString()).append('\n');
    }

    public static void writeProfilerInfoToCSV(ProfilerGroup profilerGroup, File file) {
        final var header = !file.exists();
        try (final var writer = new BufferedWriter(new FileWriter(file, true))) {
            if (header) {
                writer.append(profilerGroup.getEntries().stream()
                        .map(ProfilerEntry::getName)
                        .collect(Collectors.joining(",")));
                writer.newLine();
            }
            writer.append(profilerGroup.getEntries().stream()
                    .map(p -> String.valueOf(p.getAverage()))
                    .collect(Collectors.joining(",")));
            writer.newLine();
        } catch (Exception e) {
            throw new RuntimeException("Failed to write CSV", e);
        }
    }

}
