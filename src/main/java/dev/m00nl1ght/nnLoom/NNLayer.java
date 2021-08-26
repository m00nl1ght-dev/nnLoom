package dev.m00nl1ght.nnLoom;

import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;
import java.util.Random;

public class NNLayer {

    private final int nodeCount;
    private final int edgeCount;
    private final FloatBuffer weights;
    private final FloatBuffer biases;

    public NNLayer(int nodeCount, int edgeCount) {
        this.nodeCount = nodeCount;
        this.edgeCount = edgeCount;
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

    public void initUniform(Random random, float wSpread, float bSpread) {
        weights.clear();
        biases.clear();

        for (int i = 0; i < weights.capacity(); i++) {
            weights.put((random.nextFloat() * 2f - 1f) * wSpread);
        }

        for (int i = 0; i < biases.capacity(); i++) {
            biases.put((random.nextFloat() * 2f - 1f) * bSpread);
        }

        weights.clear();
        biases.clear();
    }

}
