
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
        v +=  input[w + inputOffset] * weights[offset + w];
    }

    output[i] = sigmoid(v);
}

kernel void backH
(

)
{
    const int i = get_global_id(0);
    const int offset = i * inputSize;

    float delta = 0;
    for (int j = 0; j < nextnumNodes; j++) {
        delta += nextnodes[j].delta * nextnodes[j].weights[i];
    }

    delta *= devsigmoid(nodes[i].output);
    nodes[i].delta = delta;

    for (int j = 0; j != nodes[i].numberOfWeights; j++) {
        nodes[i].weights[j] -= a*delta*prevnodes[j].output;
    }
}

kernel void backO
(

)
{
    const int i = get_global_id(0);
    const int offset = i * inputSize;

    float delta = 0;

    delta = (nodes[i].output-targets[i])*devsigmoid(nodes[i].output);

    for (int j = 0; j != nodes[i].numberOfWeights; j++)
        nodes[i].weights[j] -= a*delta*prevnodes[j].output;

    nodes[i].delta=delta;
}

float inline sigmoid(float v)
{
	if (v < -100) return 0;
	if (v > 100) return 1;
	return 1 / ( 1 + exp( - x ) );
}

float inline devsigmoid(float x)
{
	return ( x * ( 1 - x ) );
}
