package dev.m00nl1ght.nnLoom.opencl;

import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.CLContextCallback;

import static dev.m00nl1ght.nnLoom.opencl.CLUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryUtil.*;

public final class CLContext {

    private static final ThreadLocal<CLContext> context = new ThreadLocal<>();
    private static final Object contextLock = new Object();

    private final CLDevice device;

    private final CLContextCallback clContextCallback;
    private long clContext;

    private CLContext(CLDevice device, CLContextCallback clContextCallback, long clContext) {
        this.device = device;
        this.clContextCallback = clContextCallback;
        this.clContext = clContext;
    }

    public static CLContext create(CLDevice device) {
        synchronized (contextLock) {
            final var oldCtx = context.get();
            if (oldCtx != null) throw new IllegalStateException("Already a CLContext on this thread");

            final var clContextCB = CLContextCallback.create((errinfo, private_info, cb, user_data)
                    -> System.out.printf("[OpenCL] %s", memUTF8(errinfo)));

            final var errBuffer = BufferUtils.createIntBuffer(1);
            final var ctxProps = BufferUtils.createPointerBuffer(3)
                    .put(CL_CONTEXT_PLATFORM)
                    .put(device.getClPlatform())
                    .put(NULL)
                    .flip();

            try {

                final var clContext = clCreateContext(ctxProps, device.get(), clContextCB, NULL, errBuffer);
                checkCLError(errBuffer);

                final var ctx = new CLContext(device, clContextCB, clContext);
                context.set(ctx);
                return ctx;

            } finally {
                ctxProps.free();
            }
        }
    }

    public static void release() {
        synchronized (contextLock) {
            final var ctx = context.get();
            if (ctx == null) throw new IllegalStateException("No CLContext on this thread");
            checkCLError(clReleaseContext(ctx.clContext));
            ctx.clContextCallback.free();
            ctx.clContext = -1;
            context.set(null);
        }
    }

    public static CLContext getCurrent() {
        synchronized (contextLock) {
            final var ctx = context.get();
            if (ctx == null) throw new IllegalStateException("No CLContext on this thread");
            return ctx;
        }
    }

    public CLDevice getDevice() {
        return device;
    }

    public long get() {
        if (clContext < 0) throw new IllegalStateException("CLContext already released");
        return clContext;
    }

    public long dev() {
        return device.get();
    }

}
