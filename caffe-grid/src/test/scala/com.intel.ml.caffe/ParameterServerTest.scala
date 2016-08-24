// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.intel.ml.caffe

import java.io.File

import org.apache.spark.{SparkConf, SparkContext}
import org.parameterserver.PSConfigKeys
import org.parameterserver.cluster.MiniPSCluster
import org.slf4j.LoggerFactory

import scala.util.Random

object ParameterServerTest {
  val log = LoggerFactory.getLogger(this.getClass)
  var sc: SparkContext = null
  var conf:  com.yahoo.ml.caffe.Config = null
  var cluster: MiniPSCluster = null

  def main(args: Array[String]) {
    System.out.println("current dir: " + System.getProperty("user.dir"))
    // init conf
    val sc_conf = new SparkConf().setAppName("Caffe-on-spark-with-ps").setMaster("local[*]")
    sc_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    sc = new SparkContext(sc_conf)

    val ROOT_PATH = {
      val fullPath = getClass.getClassLoader.getResource("log4j.properties").getPath
      fullPath.substring(0, fullPath.indexOf("caffe-grid/"))
    }
    val solver_config_path = ROOT_PATH + "caffe-grid/src/test/resources/ps-integration-test/caffenet_solver.prototxt"
    val args = Array("-conf", solver_config_path,
      "-model", "file:" + ROOT_PATH + "caffe-grid/target/model.h5",
      "-imageRoot", "file:" + ROOT_PATH + "data/images",
      "-labelFile", "file:" + ROOT_PATH + "data/images/labels.txt",
      "-clusterSize", "2",
      "-devices", "1",
      "-connection", "ethernet",
      "-train",
      "-psMasterAddr", "localhost:55778", //FIXME: reset the port to your ps mini-cluster's for test
      "-psWeightVector", "w-" + Random.nextLong().toString
    )

    // setup parameter server
    val psConf = new org.parameterserver.Configuration()
    psConf.set(PSConfigKeys.PS_CHECKPOINT_PATH_KEY, "file://" + new File("target/test/data/").getAbsolutePath())
    cluster = new MiniPSCluster.Builder(psConf).masterRpcPort(55778).numSlaves(3).build

    conf = new  com.yahoo.ml.caffe.Config(sc, args)

    //
    CaffeOnSparkWithPS.bootstrap(conf, sc, true)
    Thread.sleep(100000000)
  }
}