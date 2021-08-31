package dev.m00nl1ght.nnLoom;

import dev.m00nl1ght.nnLoom.examples.MnistExample;
import dev.m00nl1ght.nnLoom.examples.XorExample;
import dev.m00nl1ght.nnLoom.opencl.CLContext;
import dev.m00nl1ght.nnLoom.opencl.CLDevice;
import dev.m00nl1ght.clockwork.profiler.DebugProfiler;
import dev.m00nl1ght.clockwork.profiler.DebugUtils;

import java.util.Random;

import static org.lwjgl.opencl.CL10.*;

public class Main {

    private Main() {}

    public static void main(String[] args) {

        final var argExample = args.length > 0 ? args[0] : "mnist";
        final var argPlatform = args.length > 1 ? args[1] : "opencl2d";
        final var argSeed = args.length > 2 ? args[2] : null;

        final var devs = CLDevice.getAvailableDevices(CL_DEVICE_TYPE_GPU);
        final var initSeed = argSeed == null ? new Random().nextLong() : argSeed.hashCode();

        for (CLDevice device : devs) {

            System.out.println("--------------------------------------------------------------");
            System.out.println("OpenCL Platform name: " + device.getPlatformInfo(CL_PLATFORM_NAME));
            System.out.println("OpenCL Device name: " + device.getDeviceInfo(CL_DEVICE_NAME));
            System.out.println("--------------------------------------------------------------");

            final var context = CLContext.create(device);

            final var nnPlatform = switch (argPlatform) {
                case "java" -> new NNPlatformJava();
                case "opencl" -> new NNPlatformOpenCL(context);
                case "opencl2d" -> new NNPlatformOpenCL2d(context);
                default -> throw new IllegalArgumentException("Invalid platform: " + argPlatform);
            };

            nnPlatform.init();

            final var profilerGroup = nnPlatform.attachDefaultProfilers().stream().findFirst().orElseThrow();
            final var profiler = new DebugProfiler();
            profiler.addGroup(profilerGroup);

            switch (argExample) {
                case "xor" -> XorExample.run(nnPlatform, initSeed);
                case "mnist" -> MnistExample.run(nnPlatform, initSeed);
                default -> throw new IllegalArgumentException("Invalid example: " + argExample);
            }

            System.out.println(DebugUtils.printProfilerInfo(profiler));

            nnPlatform.dispose();
            CLContext.release();

            break;

        }

    }



}
