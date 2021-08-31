package dev.m00nl1ght.clockwork.profiler;

import java.util.Collections;
import java.util.Set;

public interface Profilable<G extends ProfilerGroup> {

    default void attachProfiler(G profilerGroup) {
        if (!supportsProfilers())
            throw new UnsupportedOperationException(this.getClass().getSimpleName()
                    + ": This implementation does not support profilers");
    }

    default Set<? extends G> attachDefaultProfilers() {
        return Collections.emptySet();
    }

    default void detachAllProfilers() {}

    default boolean supportsProfilers() {
        return false;
    }

    default void begin(ProfilerEntry entry) {
        if (entry != null) entry.begin();
    }

    default void end(ProfilerEntry entry) {
        if (entry != null) entry.end();
    }

}
