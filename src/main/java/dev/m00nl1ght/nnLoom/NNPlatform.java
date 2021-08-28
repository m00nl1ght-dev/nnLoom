package dev.m00nl1ght.nnLoom;

import java.nio.FloatBuffer;

public interface NNPlatform {

    void init();

    FloatBuffer predict(NNetwork network, int inputCount, FloatBuffer input, int batchSize);

    void train(NNetwork network, int inputCount, FloatBuffer input, FloatBuffer targets, int epochs, int batchSize, float learningRate);

    float eval(FloatBuffer predicted, FloatBuffer targets, ErrorFunction function, int outputCount, int sampleCount);

    void dispose();

}
