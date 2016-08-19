// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import java.io.IOException;

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
    private final String weightVector; // global weight vector name on PS-master

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
      String weightVector) throws IOException {
        solverParameter_ = Utils.GetSolverParam(solver_conf_file);
        nodeRank = node_rank;
        if (!allocate(solver_conf_file, input_model_file, input_state_file,
          num_local_devices, cluster_size, node_rank, isTraining,
          connection_type, start_device_id))
            throw new RuntimeException("Failed to create CaffeNet object");

        if (psMasterAddr != null && !psMasterAddr.isEmpty()) {
            // TODO: Add psClient connection check
            LOG.info("PS-Master Address: " + psMasterAddr + ", weightVector"
              + ": " + weightVector + ", my rank: " + node_rank);
            this.psClient = new PSClient(psMasterAddr);
            this.weightVector = weightVector;

            // Note that each ps context is local and all clients are the same
            String psContext = "Caffe_On_Spark_PS_" + weightVector;
            LOG.info("CaffeNet: set local ps client context, rank: " + node_rank);
            psClient.setContext(psContext);

            // create global weight vector on the PS with overwrite
            if (nodeRank == 0) {
                int numFeatures = this.getLocalGradients().length; // TODO: further check
                LOG.info("CaffeNet: send create vector [" + weightVector + "] request to "
                  + "PS, size: " + numFeatures);
                psClient.createVector(weightVector, true, numFeatures, DataType.Float, true);
            }
        } else {
            LOG.error("NO PS Master socket Address is set.");
            this.psClient = null;
            this.weightVector = null;
        }
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

    public boolean trainWithPS(int solver_index, FloatBlob[] data, FloatArray labels) {
        try {
            LOG.info("trainWithPS: read weights from PS and set to local weight, rank: " + nodeRank);
            float[] weights = ((org.parameterserver.protocol.FloatArray)psClient.getVector
              (weightVector).getValues()).getValues();
            this.setLocalWeights(weights);
        } catch (IOException ioe) {
            LOG.error("PS vector client read IOException...", ioe);
        }

        LOG.info("trainWithPS: do one-step training with local caffe, rank: " + nodeRank);
        boolean succ = train(solver_index, data, labels);

        try {
            LOG.info("trainWithPS: getLocalGradients, rank: " + nodeRank);
            float[] gradients = this.getLocalGradients();
            psClient.add2Vector(weightVector, intArray(gradients.length), new org
              .parameterserver.protocol.FloatArray(gradients));
        } catch (IOException ioe) {
            LOG.error("PS vector client write IOException...", ioe);
        }

        // set local caffe weights with values fetch from ps
        LOG.info("trainWithPS: send PS-BSP sync request, rank: " + nodeRank);
        try {
            LOG.info("trainWithPS: send PS-BSP sync request, rank: " + nodeRank);
            psClient.bspSync();
        } catch (IOException ioe) {
            LOG.error("PS BSP sync exception..." + ioe);
        }

        return succ;
    }

    /**
     * Close ps client connection
     */
    public void closePS() {
        LOG.info("CaffeNet: send close ps client request");
        psClient.close();
    }

    // TODO: Add this method to TestUtils class

  /**
   * Create an int array filled with
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
