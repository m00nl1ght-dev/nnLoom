package dev.m00nl1ght.nnLoom.examples;

import dev.m00nl1ght.nnLoom.*;
import dev.m00nl1ght.nnLoom.mnist.MnistReader;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
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
                .layerFC(42, Activation.Tanh, Initialisation.Xavier)
                .layerFC(10, Activation.Tanh, Initialisation.Xavier)
                .layerFC(MNIST_CAT, Activation.Sigmoid, Initialisation.Uniform)
                .build();

        System.out.println("Initialising network with seed " + initSeed + " ...");
        network.init(new Random(initSeed));

        System.out.println("Loading MNIST ...");
        final var cl = MnistExample.class.getClassLoader();
        final var dataTr = MnistReader.readData(cl.getResourceAsStream(RES_DATA_TR));
        final var labelsTr = MnistReader.readLabels(cl.getResourceAsStream(RES_LABELS_TR));
        final var dataTe = MnistReader.readData(cl.getResourceAsStream(RES_DATA_TE));
        final var labelsTe = MnistReader.readLabels(cl.getResourceAsStream(RES_LABELS_TE));

        System.out.println("Evaluating network ...");
        eval(nnPlatform, network, dataTe, labelsTe, MNIST_SIZE_TE, 0);

        System.out.println("--------------------------------------------------------------");

        System.out.println("Training network ...");
        nnPlatform.train(network, MNIST_SIZE_TR, dataTr, labelsTr, 5, 100, 0.01f);

        System.out.println("Evaluating network ...");
        eval(nnPlatform, network, dataTe, labelsTe, MNIST_SIZE_TE, 0);

        System.out.println("--------------------------------------------------------------");

    }

    private static void eval(NNPlatform nnPlatform, NNetwork network, FloatBuffer data, FloatBuffer labels, int count, int incEx) {
        data.clear(); labels.clear();

        final var results = nnPlatform.predict(network, count, data, -1);
        final var error = nnPlatform.eval(results, labels, ErrorFunction.MSE, MNIST_CAT, count);
        results.clear(); labels.clear();

        final var incDist = new int[MNIST_CAT];
        final var incorrect = new ArrayList<Integer>();
        for (int i = 0; i < count; i++) {
            final var pred = maxResult(results, i);
            final var label = maxResult(labels, i);
            if (pred != label) {
                incorrect.add(i);
                incDist[label]++;
            }
        }

        final float incP = 100f * incorrect.size() / count;
        System.out.printf("Accuracy is now: %.2f %% %n", 100 - incP);
        System.out.printf("MSE is now: %.3f%n", error);

        if (incEx > 0) for (int i = 0; i < MNIST_CAT; i++) {
            System.out.println("Incorrect predictions in category " + i + ": " + incDist[i]);
        }

        Collections.shuffle(incorrect);
        if (incEx > 0 && !incorrect.isEmpty()) {
            System.out.println("Examples for incorrect predictions:");
            for (int i = 0; i < incEx && i < incorrect.size(); i++) {
                final var idx = incorrect.get(i);
                printImage(data, labels, idx);
                printResults(results, labels, idx);
            }
        }

        data.clear(); labels.clear();
    }

    private static void printImage(FloatBuffer data, FloatBuffer labels, int idx) {

        data.clear();
        data.position(idx * MNIST_PX * MNIST_PX);

        System.out.println("Image " + idx + " with label " + maxResult(labels, idx) + ":");
        System.out.println("#".repeat(MNIST_PX + 2));
        for (int i = 0; i < MNIST_PX; i++) {
            System.out.print("#");
            for (int j = 0; j < MNIST_PX; j++) {
                final var v = data.get();
                if (v > 0.7f) System.out.print("O");
                else if (v > 0.4f) System.out.print("-");
                else System.out.print(" ");
            }
            System.out.print("#");
            System.out.println();
        }
        System.out.println("#".repeat(MNIST_PX + 2));

        data.clear();

    }

    private static void printResults(FloatBuffer results, FloatBuffer labels, int idx) {

        System.out.println("Results for image " + idx + " with label " + maxResult(labels, idx) + ":");

        results.clear();

        for (int i = 0; i < MNIST_CAT; i++) {
            System.out.printf(i + " -> %.3f%n", results.get(idx * MNIST_CAT + i));
        }

        results.clear();
    }

    private static int maxResult(FloatBuffer data, int idx) {
        final var vals = new float[MNIST_CAT];
        data.get(idx * MNIST_CAT, vals);

        int maxAt = 0;
        for (int i = 1; i < vals.length; i++) {
            maxAt = vals[i] > vals[maxAt] ? i : maxAt;
        }

        return maxAt;
    }

}
