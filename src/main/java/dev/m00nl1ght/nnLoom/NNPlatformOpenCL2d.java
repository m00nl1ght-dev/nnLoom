package dev.m00nl1ght.nnLoom;

import dev.m00nl1ght.nnLoom.opencl.CLContext;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static dev.m00nl1ght.nnLoom.opencl.CLUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryUtil.*;

public class NNPlatformOpenCL2d implements NNPlatform {

    private final CLContext clContext;
    private final IntBuffer errBuffer;
    private final PointerBuffer workSize;

    private long clCommandQueue = -1;
    private long clProgram = -1;
    private long[] clKernelForward;
    private long[] clKernelBackH;
    private long[] clKernelBackO;
    private long clKernelApplyDeltas;

    public NNPlatformOpenCL2d(CLContext clContext) {
        this.clContext = Objects.requireNonNull(clContext);
        this.errBuffer = BufferUtils.createIntBuffer(1);
        this.workSize = BufferUtils.createPointerBuffer(2);
    }

    @Override
    public void init() {

        checkContext(false);

        clCommandQueue = clCreateCommandQueue(clContext.get(), clContext.dev(), NULL, errBuffer);
        checkCLError(errBuffer);

        final var source = ioResourceToByteBuffer("nnPlatform2d.cl", 1024);

        final var strings = BufferUtils.createPointerBuffer(1);
        final var lengths = BufferUtils.createPointerBuffer(1);

        strings.put(0, source);
        lengths.put(0, source.remaining());

        clProgram = clCreateProgramWithSource(clContext.get(), strings, lengths, errBuffer);
        checkCLError(errBuffer);

        checkCLError(clBuildProgram(clProgram, clContext.dev(), "", null, NULL));

        clKernelApplyDeltas = clCreateKernel(clProgram, "applyDeltas", errBuffer);
        checkCLError(errBuffer);

        final var actCount = Activation.values().length;
        clKernelForward = new long[actCount];
        clKernelBackH = new long[actCount];
        clKernelBackO = new long[actCount];

        for (final var act : Activation.values()) {

            clKernelForward[act.ordinal()] = clCreateKernel(clProgram, "forward" + act.name(), errBuffer);
            checkCLError(errBuffer);

            clKernelBackH[act.ordinal()] = clCreateKernel(clProgram, "backH" + act.name(), errBuffer);
            checkCLError(errBuffer);

            clKernelBackO[act.ordinal()] = clCreateKernel(clProgram, "backO" + act.name(), errBuffer);
            checkCLError(errBuffer);

        }

    }

    @Override
    public FloatBuffer predict(NNetwork network, int inputCount, FloatBuffer input, int batchSize) {

        checkContext(true);
        checkBuffer(input, inputCount * network.getInputCount());
        if (batchSize < 0) batchSize = inputCount;

        final var bfInput = createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input);
        final var bfVals = createBuffers(network, CL_MEM_READ_WRITE, NNLayer::getNodeCount, batchSize);
        final var bfWeights = createBuffers(network, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NNLayer::getWeights);
        final var bfBiases = createBuffers(network, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NNLayer::getBiases);
        final var results = BufferUtils.createFloatBuffer(network.getOutputCount() * inputCount);

        final var sTime = System.currentTimeMillis();

        var bNum = 0;
        var bRemaining = inputCount;
        while (bRemaining > 0) {

            final var bSize = Math.min(batchSize, bRemaining);
            final var bOffset = bNum * batchSize;

            feedForward(network, bfInput, bOffset, bSize, bfVals, bfWeights, bfBiases);

            results.limit(network.getOutputCount() * (bOffset + bSize));
            results.position(network.getOutputCount() * bOffset);
            final var bfResult = bfVals[network.getLayers().size() - 1];
            checkCLError(clEnqueueReadBuffer(clCommandQueue, bfResult, true, 0, results, null, null));

            bRemaining -= bSize;
            bNum++;

        }

        final var eTime = System.currentTimeMillis();
        System.out.println("Predicted for " + inputCount + " data sets in " + (eTime - sTime) + " ms.");

        checkCLError(clReleaseMemObject(bfInput));
        for (final var bf : bfVals) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfWeights) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfBiases) checkCLError(clReleaseMemObject(bf));

        results.clear();
        return results;

    }

    @Override
    public void train(NNetwork network, int inputCount, FloatBuffer input, FloatBuffer targets, int epochs, int batchSize, float learningRate) {

        checkContext(true);
        checkBuffer(input, inputCount * network.getInputCount());
        checkBuffer(targets, inputCount * network.getOutputCount());
        if (batchSize < 0) batchSize = inputCount;

        // shuffleBuffers(network, inputCount, input, targets);

        final var bfInput = createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input);
        final var bfTargets = createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, targets);
        final var bfVals = createBuffers(network, CL_MEM_READ_WRITE, NNLayer::getNodeCount, batchSize);
        final var bfDeltas = createBuffers(network, CL_MEM_READ_WRITE, NNLayer::getNodeCount, batchSize);
        final var bfWeights = createBuffers(network, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NNLayer::getWeights);
        final var bfBiases = createBuffers(network, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NNLayer::getBiases);

        final var sTime = System.currentTimeMillis();

        for (int e = 0; e < epochs; e++) {

            var bNum = 0;
            var bRemaining = inputCount;
            while (bRemaining > 0) {

                final var bSize = Math.min(batchSize, bRemaining);
                final var bOffset = bNum * batchSize;

                feedForward(network, bfInput, bOffset, bSize, bfVals, bfWeights, bfBiases);
                backProp(network, bfInput, bfTargets, bOffset, bSize, bfVals, bfDeltas, bfWeights, bfBiases, learningRate);

                bRemaining -= bSize;
                bNum++;

            }

        }

        final var eTime = System.currentTimeMillis();
        System.out.println("Completed " + epochs + " epochs using " + inputCount + " data sets in " + (eTime - sTime) + " ms.");

        updateBuffers(network, bfWeights, NNLayer::getWeights);
        updateBuffers(network, bfBiases, NNLayer::getBiases);

        checkCLError(clReleaseMemObject(bfInput));
        checkCLError(clReleaseMemObject(bfTargets));
        for (final var bf : bfVals) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfDeltas) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfWeights) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfBiases) checkCLError(clReleaseMemObject(bf));

    }

    private void feedForward(NNetwork network, long bfInput, int offset, int size, long[] bfVals, long[] bfWeights, long[] bfBiases) {

        final var layers = network.getLayers();

        for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {

            final var layer = layers.get(layerIdx);
            final var kern = clKernelForward[layer.getActivation().ordinal()];

            clSetKernelArg1i(kern, 0, layer.getNodeCount());
            clSetKernelArg1p(kern, 1, bfVals[layerIdx]);
            clSetKernelArg1p(kern, 2, bfWeights[layerIdx]);
            clSetKernelArg1p(kern, 3, bfBiases[layerIdx]);
            clSetKernelArg1i(kern, 4, layer.getEdgeCount());

            if (layerIdx == 0) {
                clSetKernelArg1i(kern, 5, offset * network.getInputCount());
                clSetKernelArg1p(kern, 6, bfInput);
            } else {
                clSetKernelArg1i(kern, 5, 0);
                clSetKernelArg1p(kern, 6, bfVals[layerIdx - 1]);
            }

            runKernel(kern, layer.getNodeCount(), size);

        }

    }

    private void backProp(NNetwork network, long bfInput, long bfTargets, int offset, int size,
                          long[] bfVals, long[] bfDeltas, long[] bfWeights, long[] bfBiases, float learningRate) {

        final var layers = network.getLayers();
        final var outputLayerIdx = layers.size() - 1;
        final var outputLayer = layers.get(outputLayerIdx);
        final var kernO = clKernelBackO[outputLayer.getActivation().ordinal()];

        clSetKernelArg1i(kernO, 0, outputLayer.getNodeCount());
        clSetKernelArg1p(kernO, 1, bfVals[outputLayerIdx]);
        clSetKernelArg1p(kernO, 2, bfDeltas[outputLayerIdx]);
        clSetKernelArg1i(kernO, 3, offset * outputLayer.getNodeCount());
        clSetKernelArg1p(kernO, 4, bfTargets);

        runKernel(kernO, outputLayer.getNodeCount(), size);

        for (int layerIdx = outputLayerIdx - 1; layerIdx >= 0; layerIdx--) {

            final var layer = layers.get(layerIdx);
            final var kern = clKernelBackH[layer.getActivation().ordinal()];

            clSetKernelArg1i(kern, 0, layer.getNodeCount());
            clSetKernelArg1p(kern, 1, bfVals[layerIdx]);
            clSetKernelArg1p(kern, 2, bfDeltas[layerIdx]);
            clSetKernelArg1i(kern, 3, layers.get(layerIdx + 1).getNodeCount());
            clSetKernelArg1p(kern, 4, bfDeltas[layerIdx + 1]);
            clSetKernelArg1p(kern, 5, bfWeights[layerIdx + 1]);

            runKernel(kern, layer.getNodeCount(), size);

        }

        for (int layerIdx = outputLayerIdx; layerIdx >= 0; layerIdx--) {

            final var layer = layers.get(layerIdx);

            clSetKernelArg1i(clKernelApplyDeltas, 0, layer.getNodeCount());
            clSetKernelArg1p(clKernelApplyDeltas, 1, bfVals[layerIdx]);
            clSetKernelArg1p(clKernelApplyDeltas, 2, bfDeltas[layerIdx]);
            clSetKernelArg1p(clKernelApplyDeltas, 3, bfWeights[layerIdx]);
            clSetKernelArg1p(clKernelApplyDeltas, 4, bfBiases[layerIdx]);
            clSetKernelArg1i(clKernelApplyDeltas, 5, layer.getEdgeCount());

            if (layerIdx == 0) {
                clSetKernelArg1i(clKernelApplyDeltas, 6, offset * network.getInputCount());
                clSetKernelArg1p(clKernelApplyDeltas, 7, bfInput);
            } else {
                clSetKernelArg1i(clKernelApplyDeltas, 6, 0);
                clSetKernelArg1p(clKernelApplyDeltas, 7, bfVals[layerIdx - 1]);
            }

            clSetKernelArg1f(clKernelApplyDeltas, 8, learningRate);

            runKernel(clKernelApplyDeltas, layer.getNodeCount(), size);

        }

    }

    private void shuffleBuffers(NNetwork network, int items, FloatBuffer dInputs, FloatBuffer dTargets) {

        final var order = IntStream.range(0, items).boxed()
                .collect(Collectors.toCollection(() -> new ArrayList<>(items)));

        Collections.shuffle(order);

        dInputs.mark();
        dTargets.mark();

        final var si = network.getInputCount();
        final var so = network.getOutputCount();
        final var sInputs = BufferUtils.createFloatBuffer(dInputs.remaining());
        final var sTargets = BufferUtils.createFloatBuffer(dTargets.remaining());
        sInputs.put(dInputs.asReadOnlyBuffer());
        sTargets.put(dTargets.asReadOnlyBuffer());
        sInputs.clear(); sTargets.clear();

        for (int i = 0; i < items; i++) {
            final var idx = order.get(i);
            for (int j = 0; j < si; j++) dInputs.put(sInputs.get());
            for (int j = 0; j < so; j++) dTargets.put(sTargets.get());
        }

        if (Stream.of(sInputs, sTargets, dInputs, dTargets).anyMatch(floatBuffer -> floatBuffer.remaining() > 0))
            throw new IllegalStateException();

        dInputs.reset();
        dTargets.reset();

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

    private long[] createBuffers(NNetwork network, long flags, ToIntFunction<NNLayer> size, int sizeMul) {
        final var buffers = new long[network.getLayers().size()];
        final var layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            buffers[i] = clCreateBuffer(clContext.get(), flags, size.applyAsInt(layers.get(i)) * 4 * sizeMul, errBuffer);
            checkCLError(errBuffer);
        }
        return buffers;
    }

    private long[] createBuffers(NNetwork network, long flags, Function<NNLayer, FloatBuffer> bf) {
        final var buffers = new long[network.getLayers().size()];
        final var layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            buffers[i] = clCreateBuffer(clContext.get(), flags, bf.apply(layers.get(i)), errBuffer);
            checkCLError(errBuffer);
        }
        return buffers;
    }

    private void updateBuffers(NNetwork network, long[] src, Function<NNLayer, FloatBuffer> bf) {
        final var layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            final var buffer = bf.apply(layers.get(i));
            buffer.clear();
            checkCLError(clEnqueueReadBuffer(clCommandQueue, src[i], true, 0, buffer, null, null));
            buffer.clear();
        }
    }

    private void runKernel(long clKernel, int itemCount0, int itemCount1) {
        workSize.put(0, itemCount0);
        workSize.put(1, itemCount1);
        checkCLError(clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, null, workSize, null, null, null));
        checkCLError(clFinish(clCommandQueue));
    }

    private void checkContext(boolean inited) {
        if (CLContext.getCurrent() != clContext) throw new IllegalStateException();
        if ((clCommandQueue == -1) == inited) throw new IllegalStateException();
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
    public void dispose() {

        checkContext(true);

        checkCLError(clReleaseCommandQueue(clCommandQueue));
        for (long k : clKernelForward) checkCLError(clReleaseKernel(k));
        for (long k : clKernelBackH) checkCLError(clReleaseKernel(k));
        for (long k : clKernelBackO) checkCLError(clReleaseKernel(k));
        checkCLError(clReleaseKernel(clKernelApplyDeltas));
        checkCLError(clReleaseProgram(clProgram));

        clCommandQueue = -1;

    }

}
