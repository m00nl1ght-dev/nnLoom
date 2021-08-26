package dev.m00nl1ght.nnLoom;

import dev.m00nl1ght.nnLoom.opencl.CLContext;
import dev.m00nl1ght.nnLoom.opencl.CLDevice;
import org.lwjgl.BufferUtils;

import java.util.Random;

import static org.lwjgl.opencl.CL10.*;

public class Main {

    private Main() {}

    public static void main(String[] args) {

        final var initSeed = new Random().nextLong();
        final var devs = CLDevice.getAvailableDevices(CL_DEVICE_TYPE_ALL);

        for (CLDevice device : devs) {

            System.out.println("--------------------------------------------------------------");
            System.out.println("Platform name: " + device.getPlatformInfo(CL_PLATFORM_NAME));
            System.out.println("Device name: " + device.getDeviceInfo(CL_DEVICE_NAME));

            final var context = CLContext.create(device);

            final var nnPlatform = new NNPlatformOpenCL(context);
            nnPlatform.init();

            final var network = NNetwork.builder(2)
                    .layerFC(10)
                    .layerFC(1)
                    .build();

            final var initRand = new Random(initSeed);

            network.getLayers().forEach(l -> l.setBias(0f));
            network.init(initRand, 1f);

            final var inputs = BufferUtils.createFloatBuffer(8);
            inputs.put(new float[]{0, 0, 1, 0, 0, 1, 1, 1});
            inputs.clear();

            final var outputs = BufferUtils.createFloatBuffer(4);
            outputs.put(new float[]{0, 1, 1, 0});
            outputs.clear();

            final var results = nnPlatform.predict(network, 4, inputs);

            System.out.println("Result for input 0 0 -> " + results.get());
            System.out.println("Result for input 0 1 -> " + results.get());
            System.out.println("Result for input 1 0 -> " + results.get());
            System.out.println("Result for input 1 1 -> " + results.get());

            nnPlatform.train(network, 4, inputs, outputs, 1000, 0.005f);

            final var newResults = nnPlatform.predict(network, 4, inputs);

            System.out.println("New result for input 0 0 -> " + newResults.get());
            System.out.println("New result for input 0 1 -> " + newResults.get());
            System.out.println("New result for input 1 0 -> " + newResults.get());
            System.out.println("New result for input 1 1 -> " + newResults.get());

            nnPlatform.dispose();
            CLContext.release();

            //break;

        }

    }

}
