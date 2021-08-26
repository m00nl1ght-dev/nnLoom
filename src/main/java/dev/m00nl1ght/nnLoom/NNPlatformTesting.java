package dev.m00nl1ght.nnLoom;

import dev.m00nl1ght.nnLoom.opencl.CLContext;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.nio.FloatBuffer;
import java.util.Objects;

import static dev.m00nl1ght.nnLoom.opencl.CLUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryUtil.*;

public class NNPlatformTesting implements NNPlatform {

    private static final int INPUT_SIZE = 100;

    private final CLContext clContext;

    private long clCommandQueue = -1;
    private long clProgram = -1;
    private long clKernel = -1;

    public NNPlatformTesting(CLContext clContext) {
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

        clKernel = clCreateKernel(clProgram, "sum", errBuffer);
        checkCLError(errBuffer);

    }

    @Override
    public FloatBuffer predict(NNetwork network, FloatBuffer input, int inputCount) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void train(NNetwork network, FloatBuffer input, int inputCount, int epochs) {
        throw new UnsupportedOperationException();
    }

    private FloatBuffer getABuffer() {
        // Create float array from 0 to size-1.
        FloatBuffer aBuff = BufferUtils.createFloatBuffer(INPUT_SIZE);
        float[] tempData = new float[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) {
            tempData[i] = i;
        }
        aBuff.put(tempData);
        aBuff.rewind();
        return aBuff;
    }

    private FloatBuffer getBBuffer() {
        // Create float array from size-1 to 0.
        // This means that the result should be size-1 for each element.
        FloatBuffer bBuff = BufferUtils.createFloatBuffer(INPUT_SIZE);
        float[] tempData = new float[INPUT_SIZE];
        for (int j = 0, i = INPUT_SIZE - 1; j < INPUT_SIZE; j++, i--) {
            tempData[j] = i;
        }
        bBuff.put(tempData);
        bBuff.rewind();
        return bBuff;
    }

    public void test() {

        checkContext(true);
        final var errBuffer = BufferUtils.createIntBuffer(1);

        // Create OpenCL memory object containing the first buffer's list of numbers.
        final var aMemory = clCreateBuffer(clContext.get(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, getABuffer(), errBuffer);
        checkCLError(errBuffer);

        // Create OpenCL memory object containing the second buffer's list of numbers.
        final var bMemory = clCreateBuffer(clContext.get(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, getBBuffer(), errBuffer);
        checkCLError(errBuffer);

        // Remember the length argument here is in bytes. 4 bytes per float.
        final var resultMemory = clCreateBuffer(clContext.get(), CL_MEM_READ_ONLY, INPUT_SIZE * 4, errBuffer);
        checkCLError(errBuffer);

        clSetKernelArg1p(clKernel, 0, aMemory);
        clSetKernelArg1p(clKernel, 1, bMemory);
        clSetKernelArg1p(clKernel, 2, resultMemory);
        clSetKernelArg1i(clKernel, 3, INPUT_SIZE);

        final int dimensions = 1;
        // In here we put the total number of work items we want in each dimension.
        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(dimensions);
        globalWorkSize.put(0, INPUT_SIZE);

        final var sTime = System.nanoTime();

        // Run the specified number of work units using our OpenCL program kernel.
        checkCLError(clEnqueueNDRangeKernel(clCommandQueue, clKernel, dimensions, null, globalWorkSize, null, null, null));
        checkCLError(clFinish(clCommandQueue));

        final var eTime = System.nanoTime();

        // This reads the result memory buffer.
        FloatBuffer resultBuff = BufferUtils.createFloatBuffer(INPUT_SIZE);

        // We read the buffer in blocking mode so that when the method returns we know that the result buffer is full.
        clEnqueueReadBuffer(clCommandQueue, resultMemory, true, 0, resultBuff, null, null);

        for (int i = 0; i < resultBuff.capacity(); i++) {
            if (resultBuff.get(i) != 99f) throw new RuntimeException("Incorrect value at idx " + i);
        }

        System.out.println("All values were correct! Took " + (eTime - sTime) + " nanos.");
        System.out.println("--------------------------------------------");

        checkCLError(clReleaseMemObject(aMemory));
        checkCLError(clReleaseMemObject(bMemory));
        checkCLError(clReleaseMemObject(resultMemory));

    }

    @Override
    public void dispose() {

        checkContext(true);

        checkCLError(clReleaseCommandQueue(clCommandQueue));
        checkCLError(clReleaseKernel(clKernel));
        checkCLError(clReleaseProgram(clProgram));

        clCommandQueue = -1;

    }

    private void checkContext(boolean inited) {
        if (CLContext.getCurrent() != clContext) throw new IllegalStateException();
        if ((clCommandQueue == -1) == inited) throw new IllegalStateException();
    }

}
