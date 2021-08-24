package dev.m00nl1ght.nnLoom.opencl;

import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.system.MemoryStack;

import java.nio.IntBuffer;
import java.util.HashSet;
import java.util.Set;

import static dev.m00nl1ght.nnLoom.opencl.CLUtil.*;
import static org.lwjgl.opencl.CL10.*;

public final class CLDevice {

    private static final long[] EMPTY_PTRS = new long[0];

    private final long clPlatform;
    private final long clDevice;

    private final CLCapabilities platformCapabilities;
    private final CLCapabilities deviceCapabilities;

    private CLDevice(long clPlatform, long clDevice) {
        this.clPlatform = clPlatform;
        this.clDevice = clDevice;
        this.platformCapabilities = CL.createPlatformCapabilities(clPlatform);
        this.deviceCapabilities = CL.createDeviceCapabilities(clDevice, platformCapabilities);
    }

    public static Set<CLDevice> getAvailableDevices(final int deviceType) {

        final var collected = new HashSet<CLDevice>();

        try (MemoryStack stack = MemoryStack.stackPush()) {
            final var platforms = getPlatforms(stack);
            for (final var platform : platforms) {
                final var devices = getDevices(stack, platform, deviceType);
                for (final var device : devices) {
                    collected.add(new CLDevice(platform, device));
                }
            }
        }

        return collected;

    }

    private static long[] getPlatforms(MemoryStack stack) {

        final var pi0 = stack.mallocInt(1);
        checkCLError(clGetPlatformIDs(null, pi0));
        if (pi0.get(0) == 0) return EMPTY_PTRS;

        final var platformIDs = stack.mallocPointer(pi0.get(0));
        checkCLError(clGetPlatformIDs(platformIDs, (IntBuffer) null));

        final var out = new long[platformIDs.capacity()];
        platformIDs.get(out);
        return out;

    }

    private static long[] getDevices(MemoryStack stack, long platform, int deviceType) {

        final var pi0 = stack.mallocInt(1);
        int errcode = clGetDeviceIDs(platform, deviceType, null, pi0);
        if (errcode == CL_DEVICE_NOT_FOUND) return EMPTY_PTRS;
        checkCLError(errcode);

        final var deviceIDs = stack.mallocPointer(pi0.get(0));
        checkCLError(clGetDeviceIDs(platform, deviceType, deviceIDs, (IntBuffer)null));

        final var out = new long[deviceIDs.capacity()];
        deviceIDs.get(out);
        return out;

    }

    public long getClPlatform() {
        return clPlatform;
    }

    public long get() {
        return clDevice;
    }

    public CLCapabilities getPlatformCapabilities() {
        return platformCapabilities;
    }

    public CLCapabilities getDeviceCapabilities() {
        return deviceCapabilities;
    }

    public String getPlatformInfo(final int param) {
        return getPlatformInfoStringUTF8(clPlatform, param);
    }

    public String getDeviceInfo(final int param) {
        return getDeviceInfoStringUTF8(clDevice, param);
    }

}
