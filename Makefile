HOME ?=/home/${USER}
CAFFE_ON_SPARK ?=$(shell pwd)
LD_LIBRARY_PATH ?=/home/y/lib64:/home/y/lib64/mkl/intel64
LD_LIBRARY_PATH2=${LD_LIBRARY_PATH}:${CAFFE_ON_SPARK}/caffe-public/distribute/lib:${CAFFE_ON_SPARK}/caffe-distri/distribute/lib:/usr/lib64:/lib64 
DYLD_LIBRARY_PATH ?=/home/y/lib64:/home/y/lib64/mkl/intel64
DYLD_LIBRARY_PATH2=${DYLD_LIBRARY_PATH}:${CAFFE_ON_SPARK}/caffe-public/distribute/lib:${CAFFE_ON_SPARK}/caffe-distri/distribute/lib:/usr/lib64:/lib64 

build: 
	cd caffe-public; make proto; make -j4 -e distribute; cd ..
	export LD_LIBRARY_PATH="${LD_LIBRARY_PATH2}"; mvn -B package -DskipTests
	jar -xvf caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar META-INF/native/linux64/liblmdbjni.so
	mv META-INF/native/linux64/liblmdbjni.so ${CAFFE_ON_SPARK}/caffe-distri/distribute/lib
	cp -r ${CAFFE_ON_SPARK}/caffe-public/python/caffe ${CAFFE_ON_SPARK}/caffe-grid/src/main/python/
	cd ${CAFFE_ON_SPARK}/caffe-grid/src/main/python/; zip -r caffeonsparkpythonapi  *; mv caffeonsparkpythonapi.zip ${CAFFE_ON_SPARK}/caffe-grid/target/;cd ${CAFFE_ON_SPARK}
buildosx: 
	cd caffe-public; make proto; make -j4 -e distribute; cd ..
	export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH2}"; mvn -B package
	jar -xvf caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar META-INF/native/osx64/liblmdbjni.jnilib
	mv META-INF/native/osx64/liblmdbjni.jnilib ${CAFFE_ON_SPARK}/caffe-distri/distribute/lib
	cp -r ${CAFFE_ON_SPARK}/caffe-public/python/caffe ${CAFFE_ON_SPARK}/caffe-grid/src/main/python/
	cd ${CAFFE_ON_SPARK}/caffe-grid/src/main/python/; zip -r caffeonsparkpythonapi  *; mv caffeonsparkpythonapi.zip ${CAFFE_ON_SPARK}/caffe-grid/target/; cd ${CAFFE_ON_SPARK}
update:
	git submodule init
	git submodule update --force
	git submodule foreach --recursive git clean -dfx

doc: 
	scaladoc -cp caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar:${HOME}/.m2/repository/org/apache/spark/spark-core_2.10/1.6.0/spark-core_2.10-1.6.0.jar:${HOME}/.m2/repository/org/apache/spark/spark-sql_2.10/1.6.0/spark-sql_2.10-1.6.0.jar:${HOME}/.m2/repository/org/apache/spark/spark-mllib_2.10/1.6.0/spark-mllib_2.10-1.6.0.jar:${HOME}/.m2/repository/org/apache/spark/spark-catalyst_2.10/1.6.0/spark-catalyst_2.10-1.6.0.jar:$(shell hadoop classpath)  -d scala_doc caffe-grid/src/main/scala/com/yahoo/ml/caffe/*.scala
	cd python_doc; make html; cd ..

gh-pages: 
	rm -rf scala_doc
	git checkout gh-pages scala_doc

clean: 
	cd caffe-public; make clean; cd ..
	cd caffe-distri; make clean; cd ..
	mvn clean 

ALL: build
