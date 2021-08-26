
kernel void forward
(
    global float* cVals,                // output values of current layer
    global const float* cWeights,       // weights of current layer
    const float cBias,                  // bias applied to current layer
    const int pSize,                    // number of nodes in previous layer
    const int pOffset,                  // offset for nodes in previous layer
    global const float* pVals           // output values from previous layer
)
{
    const int i = get_global_id(0);

    float v = cBias;

    for (int p = 0; p < pSize; p++) {
        v += pVals[p + pOffset] * cWeights[i * pSize + p];
    }

    cVals[i] = ( 1 / ( 1 + exp( - v ) ) );
}

kernel void backH
(
    const int cSize,                    // number of nodes in current layer
    global const float* cVals,          // values for current layer
    global float* cDeltas,              // delta values for current layer
    global float* cWeights,             // weights for current layer
    const int nSize,                    // number of nodes in next layer
    global const float* nDeltas,        // delta values for next layer
    global const float* nWeights,       // weights for next layer
    const int pSize,                    // number of nodes in previous layer
    const int pOffset,                  // offset for nodes in previous layer
    global const float* pVals,          // values for previous layer
    const float lr                      // multiplier for weight changes
)
{
    const int i = get_global_id(0);

    float delta = 0;

    for (int n = 0; n < nSize; n++) {
        delta += nDeltas[n] * nWeights[n * nSize + i];
    }

    delta *= ( cVals[i] * ( 1 - cVals[i] ) );
    cDeltas[i] = delta;

    for (int p = 0; p < pSize; p++) {
        cWeights[i * pSize + p] -= lr * delta * pVals[p + pOffset];
    }
}

kernel void backO
(
    const int cSize,                    // number of nodes in current layer
    global const float* cVals,          // values for current layer
    global float* cDeltas,              // delta values for current layer
    global float* cWeights,             // weights for current layer
    const int pSize,                    // number of nodes in previous layer
    const int pOffset,                  // offset for nodes in previous layer
    global const float* pVals,          // values for previous layer
    const int tOffset,                  // offset for target values
    global const float* tVals,          // target values from training data
    const float lr                      // multiplier for weight changes
)
{
    const int i = get_global_id(0);

    float delta = (cVals[i] - tVals[i + tOffset]) * ( cVals[i] * ( 1 - cVals[i] ) );
    cDeltas[i] = delta;

    for (int p = 0; p < pSize; p++) {
        cWeights[i * pSize + p] -= lr * delta * pVals[p + pOffset];
    }
}
