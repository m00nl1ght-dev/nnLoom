package dev.m00nl1ght.nnLoom;

import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

public class NNLayer {

    private final int nodeCount;
    private final int edgeCount;
    private final FloatBuffer weights;

    private float bias = 0.1f;

    public NNLayer(int nodeCount, int edgeCount) {
        this.nodeCount = nodeCount;
        this.edgeCount = edgeCount;
        this.weights = BufferUtils.createFloatBuffer(nodeCount * edgeCount);
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

    public float getBias() {
        return bias;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public void init(float spread) {
        weights.clear();
        for (int i = 0; i < weights.capacity(); i++) {
            //weights.put((float) (Math.random() * 2f - 1f) * spread);
            weights.put(1f);
        }
        weights.clear();
    }

}
