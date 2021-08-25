package dev.m00nl1ght.nnLoom;

import java.nio.FloatBuffer;

public interface NNPlatform {

    void init();

    FloatBuffer eval(NNetwork network, FloatBuffer input, int inputCount);

    void train(NNetwork network);

    void dispose();

}
