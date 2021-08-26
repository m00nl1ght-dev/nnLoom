package dev.m00nl1ght.nnLoom;

import dev.m00nl1ght.nnLoom.opencl.CLContext;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Objects;

import static dev.m00nl1ght.nnLoom.opencl.CLUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryUtil.*;

public class NNPlatformOpenCL implements NNPlatform {

    private final CLContext clContext;

    private long clCommandQueue = -1;
    private long clProgram = -1;
    private long clKernelForward = -1;
    private long clKernelBackH = -1;
    private long clKernelBackO = -1;

    public NNPlatformOpenCL(CLContext clContext) {
        this.clContext = Objects.requireNonNull(clContext);
    }

    @Override
    public void init() {

        checkContext(false);
        final var errBuffer = BufferUtils.createIntBuffer(1);

        clCommandQueue = clCreateCommandQueue(clContext.get(), clContext.getDevice().get(), NULL, errBuffer);
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
    public FloatBuffer predict(NNetwork network, FloatBuffer input, int inputCount) {

        checkContext(true);
        final var errBuffer = BufferUtils.createIntBuffer(1);

        final var bfInput = clCreateBuffer(clContext.get(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, input, errBuffer);
        checkCLError(errBuffer);

        final var bfVals = createValBuffers(network);
        final var bfWeights = createWeightBuffers(network);

        final var results = BufferUtils.createFloatBuffer(network.getOutputCount() * inputCount);

        final var sTime = System.nanoTime();

        for (int inputIdx = 0; inputIdx < inputCount; inputIdx++) {

            feedForward(network, bfInput, inputIdx, bfVals, bfWeights);

            results.limit(network.getOutputCount() * (inputIdx + 1) - 1);
            results.position(network.getOutputCount() * inputIdx);
            final var bfResult = bfVals[network.getLayers().size() - 1];
            clEnqueueReadBuffer(clCommandQueue, bfResult, true, 0, results, null, null);

        }

        final var eTime = System.nanoTime();

        System.out.println("Evaluated " + inputCount + " input data sets in " + (eTime - sTime) + " nanos.");

        checkCLError(clReleaseMemObject(bfInput));
        for (final var bf : bfVals) checkCLError(clReleaseMemObject(bf));
        for (final var bf : bfWeights) checkCLError(clReleaseMemObject(bf));

        results.clear();
        return results;

    }

    private void feedForward(NNetwork network, long bfInput, int inputIdx, long[] bfVals, long[] bfWeights) {

        final var layers = network.getLayers();
        final var globalWorkSize = BufferUtils.createPointerBuffer(1);

        for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {

            final var layer = layers.get(layerIdx);

            if (layerIdx == 0) {
                clSetKernelArg1p(clKernelForward, 0, bfInput);
                clSetKernelArg1i(clKernelForward, 1, inputIdx * network.getInputCount());
            } else {
                clSetKernelArg1p(clKernelForward, 0, bfVals[layerIdx - 1]);
                clSetKernelArg1i(clKernelForward, 1, 0);
            }

            clSetKernelArg1i(clKernelForward, 2, layer.getEdgeCount());
            clSetKernelArg1p(clKernelForward, 3, bfWeights[layerIdx]);
            clSetKernelArg1p(clKernelForward, 4, bfVals[layerIdx]);
            clSetKernelArg1f(clKernelForward, 5, layer.getBias());
            globalWorkSize.put(0, layer.getNodeCount());

            checkCLError(clEnqueueNDRangeKernel(clCommandQueue, clKernelForward, 1, null, globalWorkSize, null, null, null));
            checkCLError(clFinish(clCommandQueue));

        }

    }

    private long[] createValBuffers(NNetwork network) {
        final var buffers = new long[network.getLayers().size()];
        final var errBuffer = BufferUtils.createIntBuffer(1);
        List<NNLayer> layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            buffers[i] = clCreateBuffer(clContext.get(), CL_MEM_READ_ONLY, layers.get(i).getNodeCount() * 4, errBuffer);
            checkCLError(errBuffer);
        }
        return buffers;
    }

    private long[] createWeightBuffers(NNetwork network) {
        final var buffers = new long[network.getLayers().size()];
        final var errBuffer = BufferUtils.createIntBuffer(1);
        List<NNLayer> layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            buffers[i] = clCreateBuffer(clContext.get(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, layers.get(i).getWeights(), errBuffer);
            checkCLError(errBuffer);
        }
        return buffers;
    }

    @Override
    public void train(NNetwork network, FloatBuffer input, int inputCount, int epochs) {

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

    private void checkContext(boolean inited) {
        if (CLContext.getCurrent() != clContext) throw new IllegalStateException();
        if ((clCommandQueue == -1) == inited) throw new IllegalStateException();
    }

}
