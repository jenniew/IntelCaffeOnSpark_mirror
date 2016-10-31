// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import caffe.Caffe.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.protobuf.TextFormat;

import org.parameterserver.client.PSClient;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;
import static org.testng.Assert.*;

public class CaffeNetTest {
    String rootPath, solver_config_path, imagePath;
    CaffeNet net, test_net;
    SolverParameter solver_param;
    int index = 0;
    List<String> file_list;

    final int batch_size = 5;
    final int channels = 3;
    final int height = 227;
    final int width = 227;

    String psMasterAddr = "localhost:60987"; //TODO: To check that PS is running
    String weightVector = null;

    @BeforeMethod
    public void setUp() throws Exception {
        String fullPath = getClass().getClassLoader().getResource("log4j.properties").getPath();
        rootPath = fullPath.substring(0, fullPath.indexOf("caffe-distri/"));
        solver_config_path = rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt";
        net = new CaffeNet(solver_config_path,
                "",
                "",
                1, //num_local_devices,
                2, //cluster_size,
                0, //node_rank,
                true, //isTraining,
                CaffeNet.SOCKET, //SOCKET
                -1);
        assertTrue(net != null);

        test_net = new CaffeNet(solver_config_path,
                "",
                "",
                1, //num_local_devices,
                1, //cluster_size,
                0, //node_rank,
                false, //isTraining,
                0, //NONE
                -1);
        assertTrue(test_net != null);

        solver_param = Utils.GetSolverParam(solver_config_path);
        assertEquals(solver_param.getSolverMode(), SolverParameter.SolverMode.CPU);

        imagePath = rootPath + "data/images";
        file_list = Files.readAllLines(Paths.get(imagePath + "/labels.txt"), StandardCharsets.UTF_8);
    }

    @AfterMethod
    public void tearDown() throws Exception {
        net.deallocate();
    }

    @Test
    public void testBasic() {
        String[] addrs = net.localAddresses();
        assertEquals(addrs.length, 0);

        assertTrue(net.connect(addrs));

        assertTrue(net.sync());

        assertEquals(net.deviceID(0), 0);

        assertTrue(net.init(0, true));

        int from_iter = net.getInitIter(0);
        assertEquals(from_iter, 0);

        int max_iter = net.getMaxIter(0);
        assertEquals(max_iter, solver_param.getMaxIter());

        int iterId = net.snapshot();
        assertTrue(iterId >= 0);

        String state_snapshot_fn = net.snapshotFilename(0, true);
        assertTrue(state_snapshot_fn.startsWith(solver_param.getSnapshotPrefix() + "_iter_0.solverstate"));

        String model_snapshot_fn = net.snapshotFilename(0, false);
        assertTrue(model_snapshot_fn.startsWith(solver_param.getSnapshotPrefix() + "_iter_0.caffemodel"));

        String testOutputBlobNames = test_net.getTestOutputBlobNames();
        assertTrue(testOutputBlobNames.equals("accuracy,loss"));
    }

    private void nextBatch(MatVector matVec, FloatBlob labels) throws Exception {
        FloatArray labelCPU = labels.cpu_data();
        byte[] buf = new byte[1024 * 1024];

        for (int idx=0; idx<batch_size; idx ++) {
            String line = file_list.get(index++);
            if (index >= file_list.size()) index = 0;

            String[] line_splits = line.split(" ");
            String filename = line_splits[0];
            int label = Integer.parseInt(line_splits[1]);
            labelCPU.set(idx, label);

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            DataInputStream in = new DataInputStream(new FileInputStream(imagePath + "/" + filename));
            int len = in.read(buf, 0, buf.length);
            while (len > 0) {
                out.write(buf, 0, len);
                len = in.read(buf, 0, buf.length);
            }
            in.close();

            byte[] data = out.toByteArray();

            Mat mat = new Mat(data);
            mat.decode(Mat.CV_LOAD_IMAGE_COLOR);
            mat.resize(height, width);

            Mat oldmat=matVec.put(idx, mat);
            if(oldmat != null)
                oldmat.deallocate();

            out.close();
        }
    }

    @Test
    public void testMultiNetTrainWithPS() throws Exception{
        final AtomicInteger counter = new AtomicInteger(0);
        class netThread implements Runnable {
            private CaffeNet net;
            public netThread(CaffeNet net) {
                this.net = net;
            }
            @Override
            public void run() {
                try {
                    nStepsTrain(3, this.net, null, true);
                    counter.incrementAndGet();
                } catch (Exception ioe) {
                    ioe.printStackTrace();
                }
            } // end of run
        }
        this.weightVector = "w-" + (new Random()).nextLong();
        int cluster_size = 3;
        PSClient psClient = new PSClient(psMasterAddr);
        psClient.setContext("Caffe_On_Spark_PS_" + weightVector);
        psClient.bspInitializeContext(cluster_size);

        CaffeNet net1 = new CaffeNet(solver_config_path, "", "", 1, cluster_size, 0, true,
          CaffeNet.SOCKET, -1, psMasterAddr, weightVector);
        CaffeNet net2 = new CaffeNet(solver_config_path, "", "", 1, cluster_size, 1, true,
          CaffeNet.SOCKET, -1, psMasterAddr, weightVector);
        CaffeNet net3 = new CaffeNet(solver_config_path, "", "", 1, cluster_size, 2, true,
          CaffeNet.SOCKET, -1, psMasterAddr, weightVector);

        assertTrue(net1 != null && net2 != null && net3 != null);
//        assertTrue(net1 != null && net2 != null);

        Thread t1 = new Thread(new netThread(net1));
        Thread t2 = new Thread(new netThread(net2));
        Thread t3 = new Thread(new netThread(net3));

        t1.start();
        t2.start();
        t3.start();

        t1.join();
        t2.join();
        t3.join();
        assertEquals(counter.get(), cluster_size);

//        net1.deallocate();
//        net2.deallocate();
//        net3.deallocate();
    }

    @Test
    public void getParamsLengthJNITest() {
        float[] weights = net.getLocalWeights();
        long netParamsLength = net.getParamsLength();
        assertEquals(weights.length, netParamsLength);
    }

    @Test
    public void getSetWeightJNITest() throws Exception {
        float[] weights = net.getLocalWeights();

        // JNI setLocalWeights test
        float[] newWeights = randomFloatArray(weights.length);
        assertTrue(net.setLocalWeights(newWeights));
        weights = net.getLocalWeights(); // re-fetch local weights
        assertEquals(weights, newWeights);

        // One-step training, the caffe local weights should have been changed
        nStepsTrain(1, net, null, false);

        assertNotEquals(weights, net.getLocalWeights());
    }

    @Test
    public void getSetGradientsJNITest() throws Exception {
        float[] gradients = net.getLocalGradients();
        float[] newGradients = randomFloatArray(gradients.length);
        assertTrue(net.setLocalGradients(newGradients));

        gradients = net.getLocalGradients(); // re-fetch local gradients
        assertEquals(gradients, newGradients);

        // One-step training, the caffgradients weights should have been changed
        nStepsTrain(1, net, null, false);

        assertNotEquals(gradients, net.getLocalGradients());
    }

    @Test
    public void applyUpdateJNITest() throws Exception {
        // TODO: why after applyupdate() the original gradients changed??? @shiqing
        float[] gradients = net.getLocalGradients();
        assertTrue(net.applyUpdate());
        float[] newGradients = net.getLocalGradients();
        //gradients = net.getLocalGradients(); // re-fetch local gradients
        //assertEquals(gradients, newGradients);

        //assertNotEquals(gradients, net.getLocalGradients());
    }

    @Test
    public void getGradientJNITest() throws Exception {
        double momentum = 0.9;
        double base_lr = 0.001;
        double weight_decay = 0.0005;
        // Get weights from local caffe before training
        float[] weights = new float[net.getLocalWeights().length];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) 1.0;
        }
        net.setLocalWeights(weights);

        float[] gradients, gradients2;

        // Get gradients from local caffe test
        gradients = net.getLocalGradients();

        // One-step training from initial weights and re-fetch local gradients
        nStepsTrain(1, net, null, false);

        gradients2 = net.getLocalGradients();
        float[] newWeights = net.getLocalWeights();

        float[] expectedNewWeight = new float[weights.length];
        for (int i = 0; i < weights.length; i++) {
            expectedNewWeight[i] = (float) (weights[i] + (momentum * gradients[i]
              - base_lr * (gradients2[i] + weight_decay * weights[i])));
        }

        assertEquals(gradients.length, gradients2.length);
        // TODO: how to test gradients? (@shiqing) Then comment me back
//        org.junit.Assert.assertArrayEquals(expectedNewWeight, newWeights, (float) 5e-7);
        assertEquals(weights.length, gradients.length);
    }

    @Test
    public void testTrain() throws Exception {
        int batchs = 5;
        nStepsTrain(batchs, net, test_net, false);
    }

  /**
   * Do # steps of local training
   *
   * @param steps # steps of local training
   * @param trainNet caffeNet instance for interacting with Caffe (c++)
   * @param testNet if not null, also do # steps of simplified caffenet testing
   * @param withPS if true, use parameter server for parameter synchronization
   * @throws Exception
   */
    private void nStepsTrain(int steps, CaffeNet trainNet, CaffeNet testNet, boolean withPS) throws Exception {
        SolverParameter solver_param = Utils.GetSolverParam(rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt");

        String net_proto_file = solver_param.getNet();
        NetParameter net_param = Utils.GetNetParam(rootPath + "caffe-distri/" + net_proto_file);

        //blob
        MatVector matVec = new MatVector(batch_size);
        FloatBlob[] dataBlobs = new FloatBlob[1];
        FloatBlob data_blob = new FloatBlob();
        data_blob.reshape(batch_size, channels, height, width);
        dataBlobs[0] = data_blob;

        FloatBlob labelblob = new FloatBlob();
        labelblob.reshape(batch_size, 1, 1, 1);

        //transformer
        LayerParameter train_layer_param = net_param.getLayer(0);
        TransformationParameter param = train_layer_param.getTransformParam();
        FloatDataTransformer xform = new FloatDataTransformer(param, true);

        //simplified training
        System.out.print("CaffeNetTest training:");
        for (int i = 0; i < steps; i++) {
            System.out.print(".");
            nextBatch(matVec, labelblob);
            xform.transform(matVec, data_blob);
            if (withPS)
                assertTrue(trainNet.trainWithPS(0, dataBlobs, labelblob.cpu_data()));
            else
                assertTrue(trainNet.train(0, dataBlobs, labelblob.cpu_data()));
        }

        if (testNet != null) {
            //simplified test
            String[] test_features = {"loss"};
            System.out.print("CaffeNetTest test:");
            for (int i = 0; i < steps; i++) {
                System.out.print(".");
                nextBatch(matVec, labelblob);
                xform.transform(matVec, data_blob);
                FloatBlob[] top_blobs_vec =
                  testNet.predict(0, dataBlobs, labelblob.cpu_data(), test_features);
                //validate test results
                for (int j = 0; j < top_blobs_vec.length; j++) {
                    FloatArray result_vec = top_blobs_vec[j].cpu_data();
                    assertTrue(result_vec.get(0) < 50.0);
                }
            }
        }

        //release C++ resource
        xform.deallocate();
        data_blob.deallocate();
        matVec.deallocate();
    }

    /**
     * Generate random array filled with float number from (0, 1) with length len.
     * 
     * @param len array length
     * @return
     */
    public static float[] randomFloatArray(int len) {
        assert (len > 0);
        float[] result = new float[len];
        for (int i = 0;i < len; i++) {
            result[i] = (float) Math.random();
        }
        return result;
    }

}
