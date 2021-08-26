
kernel void sum
(
    global const float* a,
    global const float* b,
    global float* result,
    int const size
)
{
    const int itemId = get_global_id(0);

    if (itemId < size) {
        result[itemId] = a[itemId] + b[itemId];
    }
}

kernel void forward
(
    global const float* input,          // output values from previous layer
    const int inputOffset,              // offset for input buffer
    const int inputSize,                // number of nodes in previous layer
    global const float* weights,        // input weights of this layer
    global float* output,               // buffer for resulting output values
    const float bias                    // bias applied to input for this layer
)
{
    const int i = get_global_id(0);
    const int offset = i * inputSize;

    float v = bias;

    for (int w = 0; w < inputSize; w++) {
        v += input[w + inputOffset] * weights[offset + w];
    }

    output[i] = ( 1 / ( 1 + exp( - v ) ) );
}

kernel void backH
(
    const int cSize,                     // number of nodes in current layer
    global float* cWeights,              // weights for current layer
    global float* cDeltas,               // delta values for current layer
    global const float* cVals,           // values for current layer
    const int nSize,                     // number of nodes in next layer
    global const float* nWeights,        // weights for next layer
    global const float* nDeltas,         // delta values for next layer
    const int pSize,                     // number of nodes in previous layer
    global const float* pVals,           // values for previous layer
    const float mpx                      // multiplier for weight changes
)
{
    const int i = get_global_id(0);

    float delta = 0;

    for (int j = 0; j < nSize; j++) {
        delta += nDeltas[j] * nWeights[j * nSize + i];
    }

    delta *= ( cVals[i] * ( 1 - cVals[i] ) );
    cDeltas[i] = delta;

    for (int j = 0; j < pSize; j++) {
        cWeights[i * cSize + j] -= mpx * delta * pVals[j];
    }
}

kernel void backO
(
    const int cSize,                     // number of nodes in current layer
    global float* cWeights,              // weights for current layer
    global float* cDeltas,               // delta values for current layer
    global const float* cVals,           // values for current layer
    const int pSize,                     // number of nodes in previous layer
    global const float* pVals,           // values for previous layer
    global const float* tVals,           // target values from training data
    const float mpx                      // multiplier for weight changes
)
{
    const int i = get_global_id(0);

    float delta = (cVals[i] - tVals[i]) * ( cVals[i] * ( 1 - cVals[i] ) );
    cDeltas[i] = delta;

    for (int j = 0; j < pSize; j++) {
        cWeights[i * cSize + j] -= mpx * delta * pVals[j];
    }
}
