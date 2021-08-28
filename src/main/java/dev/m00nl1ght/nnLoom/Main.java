package dev.m00nl1ght.nnLoom;

import dev.m00nl1ght.nnLoom.examples.MnistExample;
import dev.m00nl1ght.nnLoom.opencl.CLContext;
import dev.m00nl1ght.nnLoom.opencl.CLDevice;

import static org.lwjgl.opencl.CL10.*;

public class Main {

    private Main() {}

    public static void main(String[] args) {

        final var initSeed = new Random().nextLong();
        final var devs = CLDevice.getAvailableDevices(CL_DEVICE_TYPE_CPU);

        for (CLDevice device : devs) {

            System.out.println("--------------------------------------------------------------");
            System.out.println("Platform name: " + device.getPlatformInfo(CL_PLATFORM_NAME));
            System.out.println("Device name: " + device.getDeviceInfo(CL_DEVICE_NAME));

            final var context = CLContext.create(device);

            final var nnPlatform = new NNPlatformOpenCL2d(context);
            nnPlatform.init();

            // XorExample.run(nnPlatform, initSeed);
            MnistExample.run(nnPlatform, initSeed);

            nnPlatform.dispose();
            CLContext.release();

            break;

        }

    }



}
