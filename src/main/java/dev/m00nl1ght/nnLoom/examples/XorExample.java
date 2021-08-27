package dev.m00nl1ght.nnLoom.examples;

import dev.m00nl1ght.nnLoom.Activation;
import dev.m00nl1ght.nnLoom.Initialisation;
import dev.m00nl1ght.nnLoom.NNPlatform;
import dev.m00nl1ght.nnLoom.NNetwork;
import org.lwjgl.BufferUtils;

import java.util.Random;

public class XorExample {

    private XorExample() {}

    public static void run(NNPlatform nnPlatform, long initSeed) {

        final var network = NNetwork.builder(2)
                .layerFC(10, Activation.Tanh, Initialisation.Xavier)
                .layerFC(5, Activation.Tanh, Initialisation.Xavier)
                .layerFC(1, Activation.Sigmoid, Initialisation.Uniform)
                .build();

        network.init(new Random(initSeed));

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

        nnPlatform.train(network, 4, inputs, outputs, 10000, 0.1f);

        final var newResults = nnPlatform.predict(network, 4, inputs);

        System.out.println("New result for input 0 0 -> " + newResults.get());
        System.out.println("New result for input 0 1 -> " + newResults.get());
        System.out.println("New result for input 1 0 -> " + newResults.get());
        System.out.println("New result for input 1 1 -> " + newResults.get());

    }

}
