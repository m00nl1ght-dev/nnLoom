package dev.m00nl1ght.nnLoom;

import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

public class NNPlatformJava implements NNPlatform {

    @Override
    public void init() {}

    @Override
    public FloatBuffer predict(NNetwork network, int inputCount, FloatBuffer input, int batchSize) {

        checkBuffer(input, inputCount * network.getInputCount());
        input = input.asReadOnlyBuffer();

        final var vals = network.getLayers().stream().map(l -> BufferUtils.createFloatBuffer(l.getNodeCount())).toArray(FloatBuffer[]::new);
        final var weights = network.getLayers().stream().map(NNLayer::getWeights).toArray(FloatBuffer[]::new);
        final var biases = network.getLayers().stream().map(NNLayer::getBiases).toArray(FloatBuffer[]::new);
        final var results = BufferUtils.createFloatBuffer(network.getOutputCount() * inputCount);

        final var sTime = System.currentTimeMillis();

        for (int inputIdx = 0; inputIdx < inputCount; inputIdx++) {
            feedForward(network, inputIdx, input, vals, weights, biases);
            results.put(vals[network.getLayers().size() - 1].asReadOnlyBuffer());
        }

        final var eTime = System.currentTimeMillis();
        System.out.println("Predicted for " + inputCount + " data sets in " + (eTime - sTime) + " ms.");

        results.clear();
        return results;

    }

    @Override
    public void train(NNetwork network, int inputCount, FloatBuffer input, FloatBuffer targets, int epochs, int batchSize, float learningRate) {

        checkBuffer(input, inputCount * network.getInputCount());
        checkBuffer(targets, inputCount * network.getOutputCount());
        input = input.asReadOnlyBuffer();
        targets = targets.asReadOnlyBuffer();

        final var vals = network.getLayers().stream().map(l -> BufferUtils.createFloatBuffer(l.getNodeCount())).toArray(FloatBuffer[]::new);
        final var deltas = network.getLayers().stream().map(l -> BufferUtils.createFloatBuffer(l.getNodeCount())).toArray(FloatBuffer[]::new);
        final var weights = network.getLayers().stream().map(NNLayer::getWeights).toArray(FloatBuffer[]::new);
        final var biases = network.getLayers().stream().map(NNLayer::getBiases).toArray(FloatBuffer[]::new);

        final var sTime = System.currentTimeMillis();

        for (int e = 0; e < epochs; e++) {
            for (int inputIdx = 0; inputIdx < inputCount; inputIdx++) {
                feedForward(network, inputIdx, input, vals, weights, biases);
                backProp(network, inputIdx, input, targets, vals, deltas, weights, biases, learningRate);
            }
        }

        final var eTime = System.currentTimeMillis();
        System.out.println("Completed " + epochs + " epochs using " + inputCount + " data sets in " + (eTime - sTime) + " ms.");

    }

    private void feedForward(NNetwork network, int inputIdx, FloatBuffer input, FloatBuffer[] vals,
                             FloatBuffer[] weights, FloatBuffer[] biases) {

        final var layers = network.getLayers();
        for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {

            final var layer = layers.get(layerIdx);
            final var cSize = layer.getNodeCount();
            final var pSize = layer.getEdgeCount();
            final var cVals = vals[layerIdx];
            final var cWeights = weights[layerIdx];
            final var cBiases = biases[layerIdx];
            final var pVals = layerIdx == 0 ? input : vals[layerIdx - 1];
            final var pOffset = layerIdx == 0 ? inputIdx * network.getInputCount() : 0;

            for (int c = 0; c < cSize; c++) {

                var v = cBiases.get(c);

                for (int p = 0; p < pSize; p++) {
                    v += pVals.get(p + pOffset) * cWeights.get(c * pSize + p);
                }

                cVals.put(c, (float) switch (layer.getActivation()) {
                    case Sigmoid -> ( 1 / ( 1 + Math.exp( - v ) ) );
                    case Tanh -> Math.tanh(v);
                });
            }
        }

    }

    private void backProp(NNetwork network, int inputIdx, FloatBuffer input, FloatBuffer targets, FloatBuffer[] vals,
                          FloatBuffer[] deltas, FloatBuffer[] weights, FloatBuffer[] biases, float learningRate) {

        final var layers = network.getLayers();

        {   // Output Layer
            final var layerIdx = layers.size() - 1;
            final var layer = layers.get(layerIdx);
            final var cSize = layer.getNodeCount();
            final var pSize = layer.getEdgeCount();
            final var cVals = vals[layerIdx];
            final var cDeltas = deltas[layerIdx];
            final var cWeights = weights[layerIdx];
            final var cBiases = biases[layerIdx];
            final var pVals = layerIdx == 0 ? input : vals[layerIdx - 1];
            final var pOffset = layerIdx == 0 ? inputIdx * network.getInputCount() : 0;
            final var tOffset = inputIdx * layer.getNodeCount();

            for (int c = 0; c < cSize; c++) {

                final var v = cVals.get(c);
                final var error = targets.get(c + tOffset) - v;

                final var delta = switch (layer.getActivation()) {
                    case Sigmoid -> error * ( v * ( 1 - v ) );
                    case Tanh -> error * ( 1 - v * v );
                };

                cDeltas.put(c, delta);
                cBiases.put(c, cBiases.get(c) + delta * learningRate);

                for (int p = 0; p < pSize; p++) {
                    final var idx = c * pSize + p;
                    cWeights.put(idx, cWeights.get(idx) + learningRate * delta * pVals.get(p + pOffset));
                }
            }
        }

        for (int layerIdx = layers.size() - 2; layerIdx >= 0; layerIdx--) {

            final var layer = layers.get(layerIdx);
            final var cSize = layer.getNodeCount();
            final var pSize = layer.getEdgeCount();
            final var cVals = vals[layerIdx];
            final var cDeltas = deltas[layerIdx];
            final var cWeights = weights[layerIdx];
            final var cBiases = biases[layerIdx];
            final var pVals = layerIdx == 0 ? input : vals[layerIdx - 1];
            final var pOffset = layerIdx == 0 ? inputIdx * network.getInputCount() : 0;
            final var nSize = layers.get(layerIdx + 1).getNodeCount();
            final var nDeltas = deltas[layerIdx + 1];
            final var nWeights = weights[layerIdx + 1];

            for (int c = 0; c < cSize; c++) {

                final var v = cVals.get(c);

                var error = 0f;
                for (int n = 0; n < nSize; n++) {
                    error += nDeltas.get(n) * nWeights.get(n * nSize + c);
                }

                final var delta = switch (layer.getActivation()) {
                    case Sigmoid -> error * ( v * ( 1 - v ) );
                    case Tanh -> error * ( 1 - v * v );
                };

                cDeltas.put(c, delta);
                cBiases.put(c, cBiases.get(c) + delta * learningRate);

                for (int p = 0; p < pSize; p++) {
                    final var idx = c * pSize + p;
                    cWeights.put(idx, cWeights.get(idx) + learningRate * delta * pVals.get(p + pOffset));
                }
            }

        }

    }

    private void checkBuffer(FloatBuffer buffer, int expectedSize) {
        if (buffer.remaining() < expectedSize) {
            throw new IllegalArgumentException("Remaining data in buffer not as expected ("
                    + buffer.remaining() + " != " + expectedSize + ")");
        }
    }

    @Override
    public float eval(FloatBuffer predicted, FloatBuffer targets, ErrorFunction function, int outputCount, int sampleCount) {
        switch (function) {

            case ME -> {
                var sum = 0f;

                for (int i = 0; i < sampleCount; i++) {
                    for (int j = 0; j < outputCount; j++) {
                        sum += Math.abs(targets.get() - predicted.get());
                    }
                }

                return sum / sampleCount;
            }

            case MSE -> {
                var sum = 0f;

                for (int i = 0; i < sampleCount; i++) {
                    for (int j = 0; j < outputCount; j++) {
                        final var diff = targets.get() - predicted.get();
                        sum += diff * diff;
                    }
                }

                return sum / sampleCount;
            }

            default -> throw new RuntimeException();

        }
    }

    @Override
    public void dispose() {}

}
