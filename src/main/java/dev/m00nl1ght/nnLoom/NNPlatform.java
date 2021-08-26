package dev.m00nl1ght.nnLoom;

import java.nio.FloatBuffer;

public interface NNPlatform {

    void init();

    FloatBuffer predict(NNetwork network, int inputCount, FloatBuffer input);

    void train(NNetwork network, int inputCount, FloatBuffer input, FloatBuffer targets, int epochs, float learningRate);

    void dispose();

}
