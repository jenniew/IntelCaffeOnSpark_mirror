// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

import caffe.Caffe.*;
import org.parameterserver.client.PSClient;
import org.parameterserver.protocol.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CaffeNet is the primary class for JVM layer to interact with Caffe (C++).
 * The basic usage:
 * (1) Construct a CaffeNet object with 1+ solvers
 * (2) Invoke CaffeNet.localAddresses() to get listening addresses of this CaffeNet
 * (3) Invoke CaffeNet.connect(remote_addresses) to establish connection among solvers and peers
 * (4) Invoke CaffeNet.train() or CaffeNet.predict() to perform training or test
 * (5) Invoke CaffeNet.snapshot() periodically to save training state into file systems,
 *     with file names defined as CaffeNet.snapshotFilename().
 */
public class CaffeNet extends BaseObject {
    private static final Logger LOG = LoggerFactory.getLogger(CaffeNet.class);

    public static final int NONE = 0;
    public static final int RDMA = 1;
    public static final int SOCKET = 2;
    private final SolverParameter solverParameter_;

    private final int nodeRank; // Rank of node; Indexed from 0
    // TODO: Add PSClient object release method
    // TODO: These new added filed should implements Serializable interface
    private final PSClient psClient; // Parameter Server Client
    // TODO: release these two ps-vectors (@shiqing)
    private final String globalVec; // global vector name on PS-master
    private final String localVec;  // local vector to push to PS
                                    // it's **co-partitioned** with globalVec

    private final float scale; // scale = 1 / (float) cluster_size
    private AtomicLong iterationCount = new AtomicLong(0);

    /**
     * constructor of CaffeNet.
     *
     * Solvers are constructed, and each solver will be assigned a device
     * Devices will be assigned to each solver.
     *
     * @param solver_conf_file file path for solver's configuration
     * @param input_model_file file path for model file
     * @param input_state_file file path for state file
     * @param num_local_devices     # of local devices
     * @param cluster_size     size of cluster
     * @param node_rank           my rank in the cluster
     * @param isTraining       true for training, false otherwise
     * @param connection_type  connection type among the servers
     * @param start_device_id  the start ID of device. default: -1
     */
    @Deprecated
    public CaffeNet(String solver_conf_file,
                    String input_model_file,
                    String input_state_file,
                    int num_local_devices,
                    int cluster_size,
                    int node_rank,
                    boolean isTraining,
                    int connection_type,
                    int start_device_id) throws IOException {
       this(solver_conf_file, input_model_file, input_state_file, num_local_devices,
         cluster_size, node_rank, isTraining, connection_type, start_device_id, "", "");
    }

    public CaffeNet(String solver_conf_file,
      String input_model_file,
      String input_state_file,
      int num_local_devices,
      int cluster_size,
      int node_rank,
      boolean isTraining,
      int connection_type,
      int start_device_id,
      String psMasterAddr,
      String globalVector) throws IOException {
        solverParameter_ = Utils.GetSolverParam(solver_conf_file);
        nodeRank = node_rank;
        scale = 1 / (float) cluster_size;

        if (!allocate(solver_conf_file, input_model_file, input_state_file,
          num_local_devices, cluster_size, node_rank, isTraining,
          connection_type, start_device_id))
            throw new RuntimeException("Failed to create CaffeNet object");

        if (psMasterAddr == null || psMasterAddr.isEmpty()) {
            LOG.error("NO PS Master socket Address is set.");
            this.psClient = null;
            this.globalVec = null;
            this.localVec = null;
            return ;
        }
        ///// CaffeOnSpark with parameter server
        // TODO: Add psClient connection check
        LOG.info("PS-Master Address: " + psMasterAddr + ", globalVec"
          + ": " + globalVector + ", my rank: " + node_rank);
        this.psClient = new PSClient(psMasterAddr);
        this.globalVec = globalVector;
        this.localVec = globalVec + "_" + nodeRank;

        // Note that each ps context is local and all clients are the same
        String psContext = "Caffe_On_Spark_PS_" + globalVec;
        LOG.info("set local ps client context at node with rank {}", nodeRank);
        psClient.setContext(psContext);

        if (nodeRank == 0) { // create global gradient vector on the PS with overwrite
            int numFeatures = this.getLocalGradients().length; // TODO: further check
            LOG.info("CaffeNet: send create vector request, vecor={}, size={}",
              globalVector, numFeatures);
            psClient.createVector(globalVec, true, numFeatures, DataType.Float, true);
        } else {
            boolean logged = false;
            while (!psClient.existVector(globalVector)) {
                if (!logged) {
                    logged = true;
                    LOG.info("rank={} waiting for rank=0 to create global vec", nodeRank);
                }
                // make sure that rank 0 has already create this.globalVec
            }
        }
        LOG.info("rank={} create local vector={}, co-partitioned with global vector",
          nodeRank, localVec);
        psClient.createVector(localVec, globalVec); // localVec is co-partitioned with globalVec
    }

    private native boolean allocate(String solver_conf_file,
                                    String input_model_file,
                                    String input_state_file,
                                    int num_local_devices,
                                    int cluster_size,
                                    int node_rank,
                                    boolean isTraining,
                                    int connection_type,
                                    int start_device_id);

    @Override
    protected native void deallocate(long address);

    /**
     * establish connection among solvers and cluster peers
     *
     * @param addresses Array of addresses, whose index represents rank
     * @return true if connected successfully
     */
    public native boolean connect(String[] addresses); //list of addresses, whose index represents rank

    /**
     * Alignment with all cluster members
     * @return true if successfully
     */
    public native boolean sync();

    /**
     * initialize the current thread to work with a specified solver.
     *
     * This ensure the current thread will be assigned with the right CPU/GPU mode, and the assigned device.
     * If enableNN==true, we will also set up neural network connection with input adapter of data layers.
     * 
     * For training/test threads, this method will be invoked implicitly.
     * You should invoke init(solver_index, false) in transformer threads.
     * @param solver_index     index of our solver
     * @param enableNN         should neural network be set up for training/test?
     * @return true if successful
     */
    public native boolean init(int solver_index, boolean enableNN);

    /* conveninent method for transformer initialization */
    public boolean init(int solver_index)  {
        return init(solver_index, false);
    }

    /**
     * Apply the given input data (as a array of blobs) onto the current network via the specified input blobs,
     * perform forward() and extract the output values associated with the output blob
     *
     * If this thread has not been initd, we will invoke init(solver_index, true).
     * @param solver_index index of our solver
     * @param data   array of input data to be attached to input blobs
     * @param labels   array of input labels to be attached to input blobs
     * @param output_blobnames array of output blob names
     * @return output data from the output blobs. null if failed
     */
    public native FloatBlob[] predict(int solver_index, FloatBlob[] data, FloatArray labels, String[] output_blobnames);

    /**
     * Apply the given input data to perform 1 step of training
     *
     * If this thread has not been initialize, we will invoke init(solver_index, true).
     * @param solver_index index of our solver
     * @param data   array of input data to be attached to input blobs
     * @param labels   array of input labels to be attached to input blobs
     * @return true iff succeed
     */
    @Deprecated
    public native boolean train(int solver_index, FloatBlob[] data, FloatArray labels);

    public native boolean forwardBackward(int solver_index, FloatBlob[] data, FloatArray labels);
    public native boolean applyUpdate();

    // TODO: need to print iteration here
    // TODO: we need to broadcast the init weight at the 0 iteration.
    public boolean trainWithPS(int solver_index, FloatBlob[] data, FloatArray labels) {
        iterationCount.incrementAndGet();
        LOG.info("============= iter={}, rank={} =============", iterationCount.get(), nodeRank);
        long oneStepTime = 0;
        long fbTimeStart = System.currentTimeMillis();
        forwardBackward(solver_index, data, labels);
        long fbTimeEnd = System.currentTimeMillis();
        LOG.info("forward backward time={} ms", (fbTimeEnd - fbTimeStart));
        oneStepTime += (fbTimeEnd - fbTimeStart);
        long pullTime, pushTime;
        try {
            long glgTimeStart = System.currentTimeMillis();
            float[] gradients = this.getLocalGradients();
            long glgTimeEnd = System.currentTimeMillis();
            long ggTime = glgTimeEnd - glgTimeStart;
            oneStepTime += ggTime;

            long uvTimeStart = System.currentTimeMillis();
            psClient.updateVector(localVec, new org.parameterserver.protocol.FloatArray(gradients));
            long uvTimeEnd = System.currentTimeMillis();

            long vpTimeStart = System.currentTimeMillis();
            psClient.vectorAxpby(globalVec, 1.0, localVec, 1.0);
            long vpTimeEnd = System.currentTimeMillis();

            long uvTime = uvTimeEnd - uvTimeStart;
            long vpTime = vpTimeEnd - vpTimeStart;
            pushTime = (uvTime + vpTime) + (glgTimeEnd - glgTimeStart);
            oneStepTime += pushTime;
            LOG.info("get gradient={} ms, update localVec={} ms, psClient.vectorAxpby={} ms, "
              + "total PUSH gradients time={} ms, gradients.len={}", ggTime, uvTime, vpTime,
              pushTime, gradients.length);

        } catch (IOException ioe) {
            LOG.error("PS vector client write IOException {}", ioe);
        }

        try {
            long bspSyncTimeStart = System.currentTimeMillis();
            // FIXME: The following logic is incorrect
            if (nodeRank == 0) { // scale global gradients at this step
                LOG.info("scale global vector, vector={}, scale={}", globalVec, scale);
                long scaleGVTimeStart = System.currentTimeMillis();
                psClient.vectorAxpby(globalVec, scale, localVec, 0);
                long scaleGVTimeEnd = System.currentTimeMillis();
                LOG.info("scale global vector time={} ms", (scaleGVTimeEnd - scaleGVTimeStart));
            }
            psClient.bspSync();

            long bspSyncTimeEnd = System.currentTimeMillis();
            LOG.info("PS BSP sync time={} ms", (bspSyncTimeEnd - bspSyncTimeStart));
            oneStepTime += (bspSyncTimeEnd - bspSyncTimeStart);

        } catch (IOException ioe) {
            LOG.error("PS BSP sync exception {}", ioe);
        }
        // Update local gradients from PS global gradients
        try {
            long ggTimeStart = System.currentTimeMillis();
            float[] gradients = ((org.parameterserver.protocol.FloatArray)psClient.getVector(globalVec)
              .getValues()).getValues();
            long ggTimeEnd = System.currentTimeMillis();
            long ggTime = ggTimeEnd - ggTimeStart;

            pullTime = ggTime;

            long sauTimeStart = System.currentTimeMillis();
            this.setLocalGradients(gradients);
            long sauTimeEnd = System.currentTimeMillis();
            long ssTime = sauTimeEnd - sauTimeStart;
            pullTime += ssTime;
            oneStepTime += pullTime;

            long auTimeStart = System.currentTimeMillis();
            this.applyUpdate();
            long auTimeEnd = System.currentTimeMillis();
            long auTime = auTimeEnd - auTimeStart;
            oneStepTime += auTime;
            LOG.info("psClient.getVector={} ms, set gradients={} ms, apply update={} ms, total "
              + "PULL gradient={} ms, total one-step training={} ms", ggTime, ssTime, auTime,
              pullTime, oneStepTime);
            LOG.info("===============================================");
        }  catch (IOException ioe) {
            LOG.error("PS read vector exception {}", ioe);
        }
        return true;
    }

    /**
     * Close ps client connection
     */
    public void closePS() {
        LOG.info("CaffeNet: send close ps client request");
        psClient.close();
    }

    /**
     * Get weights from local caffe.
     *
     * @return the local weights
     */
    public native float[] getLocalWeights();

    /**
     * Get gradients from local caffe.
     *
     * @return the local gradients
     */
    public native float[] getLocalGradients();

    /**
     * Set local caffe's weights.
     *
     * @param weights new one
     * @return true iff succeed
     */
    public native boolean setLocalWeights(float[] weights);

    /**
     * Set local caffe's gradients.
     *
     * @param gradients new one
     * @return true iff succeed
     */
    public native boolean setLocalGradients(float[] gradients);

    /**
     * retrieve the server address in which we will accept messages from peers in the cluster
     *
     * @return the server address
     */
    public native String[] localAddresses();

    /**
     * retrieve the device assigned to a given solver
     *
     * @param solver_index the index of a solver
     * @return device ID assiged to that solver
     */
    public native int deviceID(int solver_index);

    /**
     * number of iterations performed previously
     *
     * @param solver_index index of our solver
     * @return initial number of iteration
     */
    public native int getInitIter(int solver_index);

    /**
     * max number of iterations to be performed
     *
     * @param solver_index index of our solver
     * @return max number of iteration
     */
    public native int getMaxIter(int solver_index);

    /**
     * snapshot the model and state
     * @return iteration ID for which the snapshot was performed; -1 if failed
     */
    public native int snapshot();

    /**
     * get the test net output blob names
     * @return comma separate string of output blob names.
     */
    public native String getTestOutputBlobNames();

    /**
     * get the file name of mode or state snapshot
     * @param iter iteration ID
     * @param isState true for state, false for model
     * @return file path
     */
    public String snapshotFilename(int iter, boolean isState) {
        if (iter < 0) return null;

        StringBuilder extension;
        if (isState) {
            extension = new StringBuilder(".solverstate");
        } else {
            extension = new StringBuilder(".caffemodel");
        }
        if (solverParameter_.getSnapshotFormat() == SolverParameter.SnapshotFormat.HDF5) {
            extension.append(".h5");
        }

        return  solverParameter_.getSnapshotPrefix() + "_iter_"+ iter + extension.toString();
    }
};
