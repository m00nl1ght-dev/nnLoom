package dev.m00nl1ght.nnLoom;

import dev.m00nl1ght.nnLoom.opencl.CLContext;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Objects;
import java.util.function.ToIntFunction;

import static dev.m00nl1ght.nnLoom.opencl.CLUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryUtil.*;

public class NNPlatformOpenCL implements NNPlatform {

    private final CLContext clContext;
    private final IntBuffer errBuffer;
    private final PointerBuffer workSize;

    private long clCommandQueue = -1;
    private long clProgram = -1;
    private long clKernelForward = -1;
    private long clKernelBackH = -1;
    private long clKernelBackO = -1;

    public NNPlatformOpenCL(CLContext clContext) {
        this.clContext = Objects.requireNonNull(clContext);
        this.errBuffer = BufferUtils.createIntBuffer(1);
        this.workSize = BufferUtils.createPointerBuffer(1);
    }

    @Override
    public void init() {

        checkContext(false);

        clCommandQueue = clCreateCommandQueue(clContext.get(), clContext.dev(), NULL, errBuffer);
        checkCLError(errBuffer);

        final var source = ioResourceToByteBuffer("nnPlatform.cl", 1024);

        final var strings = BufferUtils.createPointerBuffer(1);
        final var lengths = BufferUtils.createPointerBuffer(1);

        strings.put(0, source);
        lengths.put(0, source.remaining());

        clProgram = clCreateProgramWithSource(clContext.get(), strings, lengths, errBuffer);
        checkCLError(errBuffer);

        checkCLError(clBuildProgram(clProgram, clContext.dev(), "", null, NULL));

        clKernelForward = clCreateKernel(clProgram, "forward", errBuffer);
        checkCLError(errBuffer);

        clKernelBackH = clCreateKernel(clProgram, "backH", errBuffer);
        checkCLError(errBuffer);

        clKernelBackO = clCreateKernel(clProgram, "backO", errBuffer);
        checkCLError(errBuffer);

    }

    @Override
    public FloatBuffer predict(NNetwork network, int inputCount, FloatBuffer input) {

        checkContext(true);
        checkBuffer(input, inputCount * network.getInputCount());

        final var bfInput = createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input);
        final var bfVals = createBuffers(network, CL_MEM_READ_WRITE, l -> l.getNodeCount() * 4);
        final var bfWeights = createWeightBuffers(network, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        final var results = BufferUtils.createFloatBuffer(network.getOutputCount() * inputCount);

        final var sTime = System.nanoTime();

        for (int inputIdx = 0; inputIdx < inputCount; inputIdx++) {

            feedForward(network, bfInput, inputIdx, bfVals, bfWeights);

            results.limit(network.getOutputCount() * (inputIdx + 1));
            results.position(network.getOutputCount() * inputIdx);
            final var bfResult = bfVals[network.getLayers().size() - 1];
            checkCLError(clEnqueueReadBuffer(clCommandQueue, bfResult, true, 0, results, null, null));

        }

        final var eTime = System.nanoTime();
        System.out.println("Predicted for " + inputCount + " data sets in " + (eTime - sTime) + " nanos.");

        checkCLError(clReleaseMemObject(bfInput));
        for (final var bf : bfVals) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfWeights) checkCLError(clReleaseMemObject(bf));

        results.clear();
        return results;

    }

    @Override
    public void train(NNetwork network, int inputCount, FloatBuffer input, FloatBuffer targets, int epochs, float learningRate) {

        checkContext(true);
        checkBuffer(input, inputCount * network.getInputCount());
        checkBuffer(targets, inputCount * network.getOutputCount());

        final var bfInput = createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input);
        final var bfTargets = createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, targets);
        final var bfVals = createBuffers(network, CL_MEM_READ_WRITE, l -> l.getNodeCount() * 4);
        final var bfDeltas = createBuffers(network, CL_MEM_READ_WRITE, l -> l.getNodeCount() * 4);
        final var bfWeights = createWeightBuffers(network, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

        final var sTime = System.nanoTime();

        for (int e = 0; e < epochs; e++) {
            for (int inputIdx = 0; inputIdx < inputCount; inputIdx++) {

                feedForward(network, bfInput, inputIdx, bfVals, bfWeights);

                backProp(network, bfInput, bfTargets, inputIdx, bfVals, bfDeltas, bfWeights, learningRate);

                updateWeights(network, bfWeights);

            }
        }

        final var eTime = System.nanoTime();
        System.out.println("Completed " + epochs + " epochs using " + inputCount + " data sets in " + (eTime - sTime) + " nanos.");

        checkCLError(clReleaseMemObject(bfInput));
        checkCLError(clReleaseMemObject(bfTargets));
        for (final var bf : bfVals) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfDeltas) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfWeights) checkCLError(clReleaseMemObject(bf));

    }

    private void feedForward(NNetwork network, long bfInput, int inputIdx, long[] bfVals, long[] bfWeights) {

        final var layers = network.getLayers();

        for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {

            final var layer = layers.get(layerIdx);

            clSetKernelArg1p(clKernelForward, 0, bfVals[layerIdx]);
            clSetKernelArg1p(clKernelForward, 1, bfWeights[layerIdx]);
            clSetKernelArg1f(clKernelForward, 2, layer.getBias());
            clSetKernelArg1i(clKernelForward, 3, layer.getEdgeCount());

            if (layerIdx == 0) {
                clSetKernelArg1i(clKernelForward, 4, inputIdx * network.getInputCount());
                clSetKernelArg1p(clKernelForward, 5, bfInput);
            } else {
                clSetKernelArg1i(clKernelForward, 4, 0);
                clSetKernelArg1p(clKernelForward, 5, bfVals[layerIdx - 1]);
            }

            runKernel(clKernelForward, layer.getNodeCount());

        }

    }

    private void backProp(NNetwork network, long bfInput, long bfTargets, int inputIdx,
                          long[] bfVals, long[] bfDeltas, long[] bfWeights, float learningRate) {

        final var layers = network.getLayers();
        final var outputLayerIdx = layers.size() - 1;
        final var outputLayer = layers.get(outputLayerIdx);

        clSetKernelArg1i(clKernelBackO, 0, outputLayer.getNodeCount());
        clSetKernelArg1p(clKernelBackO, 1, bfVals[outputLayerIdx]);
        clSetKernelArg1p(clKernelBackO, 2, bfDeltas[outputLayerIdx]);
        clSetKernelArg1p(clKernelBackO, 3, bfWeights[outputLayerIdx]);
        clSetKernelArg1i(clKernelBackO, 4, outputLayer.getEdgeCount());

        if (outputLayerIdx == 0) {
            clSetKernelArg1i(clKernelBackO, 5, inputIdx * network.getInputCount());
            clSetKernelArg1p(clKernelBackO, 6, bfInput);
        } else {
            clSetKernelArg1i(clKernelBackO, 5, 0);
            clSetKernelArg1p(clKernelBackO, 6, bfVals[outputLayerIdx - 1]);
        }

        clSetKernelArg1i(clKernelBackO, 7, inputIdx * outputLayer.getNodeCount());
        clSetKernelArg1p(clKernelBackO, 8, bfTargets);
        clSetKernelArg1f(clKernelBackO, 9, learningRate);

        runKernel(clKernelBackO, outputLayer.getNodeCount());

        for (int layerIdx = outputLayerIdx - 1; layerIdx >= 0; layerIdx--) {

            final var layer = layers.get(layerIdx);

            clSetKernelArg1i(clKernelBackH, 0, layer.getNodeCount());
            clSetKernelArg1p(clKernelBackH, 1, bfVals[layerIdx]);
            clSetKernelArg1p(clKernelBackH, 2, bfDeltas[layerIdx]);
            clSetKernelArg1p(clKernelBackH, 3, bfWeights[layerIdx]);
            clSetKernelArg1i(clKernelBackH, 4, layers.get(layerIdx + 1).getNodeCount());
            clSetKernelArg1p(clKernelBackH, 5, bfDeltas[layerIdx + 1]);
            clSetKernelArg1p(clKernelBackH, 6, bfWeights[layerIdx + 1]);
            clSetKernelArg1i(clKernelBackH, 7, layer.getEdgeCount());

            if (layerIdx == 0) {
                clSetKernelArg1i(clKernelBackH, 8, inputIdx * network.getInputCount());
                clSetKernelArg1p(clKernelBackH, 9, bfInput);
            } else {
                clSetKernelArg1i(clKernelBackH, 8, 0);
                clSetKernelArg1p(clKernelBackH, 9, bfVals[layerIdx - 1]);
            }

            clSetKernelArg1f(clKernelBackH, 10, learningRate);

            runKernel(clKernelBackH, layer.getNodeCount());

        }

    }

    private long createBuffer(long flags, int size) {
        final var buffer = clCreateBuffer(clContext.get(), flags, size, errBuffer);
        checkCLError(errBuffer);
        return buffer;
    }

    private long createBuffer(long flags, FloatBuffer data) {
        final var buffer = clCreateBuffer(clContext.get(), flags, data, errBuffer);
        checkCLError(errBuffer);
        return buffer;
    }

    private long[] createBuffers(NNetwork network, long flags, ToIntFunction<NNLayer> size) {
        final var buffers = new long[network.getLayers().size()];
        final var layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            buffers[i] = clCreateBuffer(clContext.get(), flags, size.applyAsInt(layers.get(i)), errBuffer);
            checkCLError(errBuffer);
        }
        return buffers;
    }

    private long[] createWeightBuffers(NNetwork network, long flags) {
        final var buffers = new long[network.getLayers().size()];
        final var layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            buffers[i] = clCreateBuffer(clContext.get(), flags, layers.get(i).getWeights(), errBuffer);
            checkCLError(errBuffer);
        }
        return buffers;
    }

    private void updateWeights(NNetwork network, long[] bfWeights) {
        final var layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            final var weights = layers.get(i).getWeights();
            weights.clear();
            checkCLError(clEnqueueReadBuffer(clCommandQueue, bfWeights[i], true, 0, weights, null, null));
            weights.clear();
        }
    }

    private void runKernel(long clKernel, int itemCount) {
        workSize.put(0, itemCount);
        checkCLError(clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, null, workSize, null, null, null));
        checkCLError(clFinish(clCommandQueue));
    }

    private void checkContext(boolean inited) {
        if (CLContext.getCurrent() != clContext) throw new IllegalStateException();
        if ((clCommandQueue == -1) == inited) throw new IllegalStateException();
    }

    private void checkBuffer(FloatBuffer buffer, int expectedSize) {
        if (buffer.remaining() != expectedSize) {
            throw new IllegalArgumentException("Remaining data in buffer not as expected ("
                    + buffer.remaining() + " != " + expectedSize + ")");
        }
    }

    @Override
    public void dispose() {

        checkContext(true);

        checkCLError(clReleaseCommandQueue(clCommandQueue));
        checkCLError(clReleaseKernel(clKernelForward));
        checkCLError(clReleaseKernel(clKernelBackH));
        checkCLError(clReleaseKernel(clKernelBackO));
        checkCLError(clReleaseProgram(clProgram));

        clCommandQueue = -1;

    }

}
