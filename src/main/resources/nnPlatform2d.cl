
// All kernels were implemented based on the equations, explainations and samples from the following sources:
// https://www.math.purdue.edu/~nwinovic/deep_learning.html
// https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
// https://www.math.purdue.edu/~nwinovic/deep_learning_optimization.html
// https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d
// http://forum.lwjgl.org/index.php?topic=6521.0
// https://github.com/rsnemmen/OpenCL-examples/blob/master/sum_array/vecAdd.c
// https://github.com/pablo-mayrgundter/freality/blob/master/ml/nn/Backprop.java
// https://github.com/LWJGL/lwjgl3/tree/master/modules/samples/src/test/java/org/lwjgl/demo/opencl

kernel void forwardSigmoid
(
    const int cSize,                    // number of nodes in current layer
    global float* cVals,                // output values of current layer
    global const float* cWeights,       // weights of current layer
    global const float* cBiases,        // biases of current layer
    const int pSize,                    // number of nodes in previous layer
    const int pOffset,                  // offset for nodes in previous layer
    global const float* pVals           // output values from previous layer
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float v = cBiases[i];
    for (int p = 0; p < pSize; p++) {
        v += pVals[p + pOffset + j * pSize] * cWeights[i * pSize + p];
    }

    cVals[i + j * cSize] = ( 1 / ( 1 + exp( - v ) ) );
}

kernel void forwardTanh
(
    const int cSize,                    // number of nodes in current layer
    global float* cVals,                // output values of current layer
    global const float* cWeights,       // weights of current layer
    global const float* cBiases,        // biases of current layer
    const int pSize,                    // number of nodes in previous layer
    const int pOffset,                  // offset for nodes in previous layer
    global const float* pVals           // output values from previous layer
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float v = cBiases[i];
    for (int p = 0; p < pSize; p++) {
        v += pVals[p + pOffset + j * pSize] * cWeights[i * pSize + p];
    }

    cVals[i + j * cSize] = tanh(v);
}

kernel void backHSigmoid
(
    const int cSize,                    // number of nodes in current layer
    global const float* cVals,          // values for current layer
    global float* cDeltas,              // delta values for current layer
    const int nSize,                    // number of nodes in next layer
    global const float* nDeltas,        // delta values for next layer
    global const float* nWeights        // weights for next layer
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int cj = i + j * cSize;
    const float v = cVals[cj];

    float error = 0;
    for (int n = 0; n < nSize; n++) {
        error += nDeltas[n + j * nSize] * nWeights[n * nSize + i];
    }

    cDeltas[cj] = error * ( v * ( 1 - v ) );
}

kernel void backHTanh
(
    const int cSize,                    // number of nodes in current layer
    global const float* cVals,          // values for current layer
    global float* cDeltas,              // delta values for current layer
    const int nSize,                    // number of nodes in next layer
    global const float* nDeltas,        // delta values for next layer
    global const float* nWeights        // weights for next layer
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int cj = i + j * cSize;
    const float v = cVals[cj];

    float error = 0;
    for (int n = 0; n < nSize; n++) {
        error += nDeltas[n + j * nSize] * nWeights[n * nSize + i];
    }

    cDeltas[cj] = error * ( 1 - v * v );
}

kernel void backOSigmoid
(
    const int cSize,                    // number of nodes in current layer
    global const float* cVals,          // values for current layer
    global float* cDeltas,              // delta values for current layer
    const int tOffset,                  // offset for target values
    global const float* tVals           // target values from training data
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int cj = i + j * cSize;
    const float v = cVals[cj];

    const float error = tVals[cj + tOffset] - v;
    cDeltas[cj] = error * ( v * ( 1 - v ) );
}

kernel void backOTanh
(
    const int cSize,                    // number of nodes in current layer
    global const float* cVals,          // values for current layer
    global float* cDeltas,              // delta values for current layer
    const int tOffset,                  // offset for target values
    global const float* tVals           // target values from training data
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int cj = i + j * cSize;
    const float v = cVals[cj];

    const float error = tVals[cj + tOffset] - v;
    cDeltas[cj] = error * ( 1 - v * v );
}

kernel void applyDeltas
(
    const int cSize,                    // number of nodes in current layer
    global const float* cVals,          // values for current layer
    global const float* cDeltas,        // delta values for current layer
    global float* cWeights,             // weights for current layer
    global float* cBiases,              // biases for current layer
    const int pSize,                    // number of nodes in previous layer
    const int pOffset,                  // offset for nodes in previous layer
    global const float* pVals,          // values for previous layer
    const float lr                      // multiplier for weight changes
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int cj = i + j * cSize;
    const float v = cVals[cj];
    const float delta = cDeltas[cj];

    cBiases[i] += delta * lr;

    for (int p = 0; p < pSize; p++) {
        const float dw = lr * delta * pVals[p + pOffset + j * pSize];
        cWeights[i * pSize + p] += dw;
    }
}
