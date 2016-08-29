// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import java.io.FileReader;
import java.io.IOException;
import caffe.Caffe.*;
import com.google.protobuf.TextFormat;

public class Utils {
    static public SolverParameter GetSolverParam(String file) throws IOException {
        SolverParameter.Builder solver_builder = SolverParameter.newBuilder();
        FileReader reader = new FileReader(file);
        TextFormat.merge(reader, solver_builder);
        reader.close();
        return solver_builder.build();
    }

    static public NetParameter GetNetParam(String file) throws IOException{
        NetParameter.Builder net_builder = NetParameter.newBuilder();
        FileReader reader = new FileReader(file);
        TextFormat.merge(reader, net_builder);
        reader.close();
        return net_builder.build();
    }

    /**
     * Scale a float array.
     *
     * @param array
     * @param scale
     * @return
     */
    public static float[] arrayScale(float[] array, double scale) {
        for (int i = 0; i < array.length;i ++) {
            array[i] *= scale;
        }
        return array;
    }

    /**
     * Create an int array filled with 0~(size-1).
     *
     * @param size
     * @return
     */
    public static int[] intArray(int size) {
        int[] data = new int[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }
        return data;
    }
}
