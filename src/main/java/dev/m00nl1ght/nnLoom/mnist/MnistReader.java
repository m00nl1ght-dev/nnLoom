package dev.m00nl1ght.nnLoom.mnist;

import org.lwjgl.BufferUtils;

import java.io.*;
import java.nio.FloatBuffer;
import java.util.Objects;

/**
 * Minimalistic reader for the mnist dataset.
 * Formats are described here: http://yann.lecun.com/exdb/mnist/
 */
public class MnistReader {

    private MnistReader() {}

    public static FloatBuffer readData(InputStream dataSource) {

        Objects.requireNonNull(dataSource);

        try {

            final var input = new DataInputStream(new BufferedInputStream(dataSource));

            final var magicNumber = input.readInt();
            final var itemCount = input.readInt();
            final var rowCount = input.readInt();
            final var colCount = input.readInt();

            final var size = itemCount * rowCount * colCount;
            final var data = BufferUtils.createFloatBuffer(size);

            for (int i = 0; i < size; i++) {
                data.put(input.readUnsignedByte() / 255f);
            }

            input.close();
            data.clear();
            return data;

        } catch (Exception e) {
            throw new RuntimeException("Failed to load mnist images", e);
        }

    }

    public static FloatBuffer readLabels(InputStream dataSource) {

        Objects.requireNonNull(dataSource);

        try {

            final var input = new DataInputStream(new BufferedInputStream(dataSource));

            int magicNumber = input.readInt();
            int labelCount = input.readInt();

            final var data = BufferUtils.createFloatBuffer(labelCount * 10);

            for (int i = 0; i < labelCount; i++) {
                final var label = input.readUnsignedByte();
                for (int j = 0; j < 10; j++) {
                    data.put(j == label ? 1f : 0f);
                }
            }

            input.close();
            data.clear();
            return data;

        } catch (Exception e) {
            throw new RuntimeException("Failed to load mnist images", e);
        }

    }
}