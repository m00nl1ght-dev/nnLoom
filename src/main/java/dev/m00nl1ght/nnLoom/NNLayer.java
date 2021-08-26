package dev.m00nl1ght.nnLoom;

import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;
import java.util.Objects;
import java.util.Random;

public class NNLayer {

    private final int nodeCount;
    private final int edgeCount;
    private final Activation activation;
    private final Initialisation initialisation;
    private final FloatBuffer weights;
    private final FloatBuffer biases;

    public NNLayer(int nodeCount, int edgeCount, Activation activation, Initialisation initialisation) {
        this.nodeCount = nodeCount;
        this.edgeCount = edgeCount;
        this.activation = Objects.requireNonNull(activation);
        this.initialisation = Objects.requireNonNull(initialisation);
        this.weights = BufferUtils.createFloatBuffer(nodeCount * edgeCount);
        this.biases = BufferUtils.createFloatBuffer(nodeCount);
    }

    public int getNodeCount() {
        return nodeCount;
    }

    public int getEdgeCount() {
        return edgeCount;
    }

    public FloatBuffer getWeights() {
        return weights;
    }

    public FloatBuffer getBiases() {
        return biases;
    }

    public Activation getActivation() {
        return activation;
    }

    public Initialisation getInitialisation() {
        return initialisation;
    }

    public void init(Random random) {

        weights.clear();
        biases.clear();

        switch (initialisation) {

            case Uniform -> {
                for (int i = 0; i < weights.capacity(); i++)
                    weights.put((random.nextFloat() * 2f - 1f));
                for (int i = 0; i < biases.capacity(); i++)
                    biases.put((random.nextFloat() * 2f - 1f));
            }

            case Xavier -> {
                final var mul = Math.sqrt(1d / edgeCount);
                for (int i = 0; i < weights.capacity(); i++)
                    weights.put((float) ((random.nextDouble() * 2f - 1f) * mul));
                for (int i = 0; i < biases.capacity(); i++)
                    biases.put((float) ((random.nextDouble() * 2f - 1f) * mul));
            }

        }

        weights.clear();
        biases.clear();

    }

}
