package dev.m00nl1ght.nnLoom.examples;

import dev.m00nl1ght.nnLoom.*;
import dev.m00nl1ght.nnLoom.mnist.MnistReader;

import java.nio.FloatBuffer;
import java.util.Random;

public class MnistExample {

    private static final int MNIST_PX = 28;
    private static final int MNIST_CAT = 10;
    private static final int MNIST_SIZE_TR = 60000;
    private static final int MNIST_SIZE_TE = 10000;

    private static final String RES_DATA_TR   = "mnist\\train-images.idx3-ubyte";
    private static final String RES_LABELS_TR = "mnist\\train-labels.idx1-ubyte";
    private static final String RES_DATA_TE   = "mnist\\t10k-images.idx3-ubyte";
    private static final String RES_LABELS_TE = "mnist\\t10k-labels.idx1-ubyte";

    private MnistExample() {}

    public static void run(NNPlatform nnPlatform, long initSeed) {

        final var network = NNetwork.builder(MNIST_PX * MNIST_PX)
                .layerFC(100, Activation.Tanh, Initialisation.Xavier)
                .layerFC(MNIST_CAT, Activation.Sigmoid, Initialisation.Uniform)
                .build();

        network.init(new Random(initSeed));

        final var cl = MnistExample.class.getClassLoader();
        final var dataTr = MnistReader.readData(cl.getResourceAsStream(RES_DATA_TR));
        final var labelsTr = MnistReader.readLabels(cl.getResourceAsStream(RES_LABELS_TR));
        final var dataTe = MnistReader.readData(cl.getResourceAsStream(RES_DATA_TE));
        final var labelsTe = MnistReader.readLabels(cl.getResourceAsStream(RES_LABELS_TE));

        eval(nnPlatform, network, dataTe, labelsTe);

        nnPlatform.train(network, MNIST_SIZE_TR / 12, dataTr, labelsTr, 1, 0.01f);
        eval(nnPlatform, network, dataTe, labelsTe);

        nnPlatform.train(network, MNIST_SIZE_TR / 6, dataTr, labelsTr, 1, 0.01f);
        eval(nnPlatform, network, dataTe, labelsTe);

        nnPlatform.train(network, MNIST_SIZE_TR / 3, dataTr, labelsTr, 1, 0.01f);
        eval(nnPlatform, network, dataTe, labelsTe);

        nnPlatform.train(network, MNIST_SIZE_TR, dataTr, labelsTr, 1, 0.01f);
        eval(nnPlatform, network, dataTe, labelsTe);

    }

    private static void eval(NNPlatform nnPlatform, NNetwork network, FloatBuffer data, FloatBuffer labels) {
        data.clear(); labels.clear();
        final var results = nnPlatform.predict(network, MNIST_SIZE_TE, data);
        final var error = nnPlatform.eval(results, labels, ErrorFunction.ME, MNIST_CAT, MNIST_SIZE_TE);
        System.out.printf("Error is now: %.3f%n", error);
        data.clear(); labels.clear();
    }

    private static void printImage(FloatBuffer data, FloatBuffer labels, int idx) {

        data.clear();
        data.position(idx * MNIST_PX * MNIST_PX);

        System.out.println("Image " + idx + " with label " + label(labels, idx) + ":");
        System.out.println("#".repeat(MNIST_PX + 2));
        for (int i = 0; i < MNIST_PX; i++) {
            System.out.print("#");
            for (int j = 0; j < MNIST_PX; j++) {
                final var v = data.get();
                System.out.print(v > 0.5f ? "O" : " ");
            }
            System.out.print("#");
            System.out.println();
        }
        System.out.println("#".repeat(MNIST_PX + 2));

        data.clear();

    }

    private static void printResults(FloatBuffer results, FloatBuffer labels, int idx) {

        System.out.println("Results for image " + idx + " with label " + label(labels, idx) + ":");

        results.clear();

        for (int i = 0; i < MNIST_CAT; i++) {
            System.out.printf(i + " -> %.3f%n", results.get());
        }

        results.clear();
    }

    private static int label(FloatBuffer labels, int idx) {
        final var labelA = new float[MNIST_CAT];
        labels.get(idx * MNIST_CAT, labelA);
        int label = 0; for (int i = 0; i < MNIST_CAT && labelA[i] < 1f; i++) label = i + 1;
        return label;
    }

}
