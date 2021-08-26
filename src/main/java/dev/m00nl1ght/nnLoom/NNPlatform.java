package dev.m00nl1ght.nnLoom;

import java.nio.FloatBuffer;

public interface NNPlatform {

    void init();

    FloatBuffer predict(NNetwork network, FloatBuffer input, int inputCount);

    void train(NNetwork network, FloatBuffer input, int inputCount, int epochs);

    void dispose();

}
