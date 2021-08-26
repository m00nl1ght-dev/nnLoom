package dev.m00nl1ght.nnLoom;

import java.util.ArrayList;
import java.util.List;

public class NNetwork {

    private final int inputCount;
    private final List<NNLayer> layers;

    private NNetwork(int inputCount, List<NNLayer> layers) {
        this.inputCount = inputCount;
        this.layers = List.copyOf(layers);
    }

    public int getInputCount() {
        return inputCount;
    }

    public List<NNLayer> getLayers() {
        return layers;
    }

    public int getOutputCount() {
        return layers.get(layers.size() - 1).getNodeCount();
    }

    public static Builder builder(int inputCount) {
        if (inputCount <= 0) throw new IllegalArgumentException();
        return new Builder(inputCount);
    }

    public static class Builder {

        private final int inputNodes;
        private final List<NNLayer> layers = new ArrayList<>();

        private Builder(int inputNodes) {
            this.inputNodes = inputNodes;
        }

        public NNetwork build() {
            if (layers.isEmpty()) throw new IllegalStateException();
            return new NNetwork(inputNodes, layers);
        }

        public Builder layerFC(int nodeCount) {
            final int edgeCount = layers.isEmpty() ? inputNodes : layers.get(layers.size() - 1).getNodeCount();
            layers.add(new NNLayer(nodeCount, edgeCount));
            return this;
        }

    }

}
