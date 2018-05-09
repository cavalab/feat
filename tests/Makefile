all: autoBP

sample: ../src/node/node.h ../src/node/nodeDx.h trigNodes arithNodes powerNodes eNodes miscNodes
	g++ -o sample ../src/node/node.h ../src/node/nodeDx.h ../src/node/nodecos.h -I /usr/include/eigen3/ -std=c++14 


autoBP: autoDiffTester.cc ../src/auto_backprop.h ../src/testMetrics.h nodes
	g++ -o autoBP autoDiffTester.cc ../src/auto_backprop.h ../src/testMetrics.h -I/usr/include/eigen3/ -std=c++14 -I/home/bill/anaconda3/include -L/home/bill/anaconda3/lib -rdynamic -lshogun -Wl,-rpath,/home/bill/anaconda3/lib

nodes: ../src/node/node.h ../src/node/nodeDx.h trigNodes arithNodes powerNodes eNodes miscNodes
	g++ -o nodes ../src/node/node.h ../src/node/nodeDx.h -I /usr/include/eigen3/ -std=c++14

trigNodes: ../src/node/nodecos.h ../src/node/nodesin.h ../src/node/nodetanh.h
	g++ -o trigNodes ../src/node/nodecos.h ../src/node/nodesin.h ../src/node/nodetanh.h -I /usr/include/eigen3/ -std=c++14

arithNodes: ../src/node/nodeadd.h ../src/node/nodesubtract.h ../src/node/nodemultiply.h ../src/node/nodedivide.h
	g++ -o arithNodes ../src/node/nodeadd.h ../src/node/nodesubtract.h ../src/node/nodemultiply.h ../src/node/nodedivide.h -I /usr/include/eigen3/ -std=c++14

powerNodes: ../src/node/nodecube.h ../src/node/nodeexponent.h ../src/node/nodesqrt.h ../src/node/nodesquare.h
	g++ -o powerNodes ../src/node/nodecube.h ../src/node/nodeexponent.h ../src/node/nodesqrt.h ../src/node/nodesquare.h -I /usr/include/eigen3/ -std=c++14

eNodes: ../src/node/nodeexponential.h ../src/node/nodelog.h
	g++ -o eNodes ../src/node/nodeexponential.h ../src/node/nodelog.h -I /usr/include/eigen3/ -std=c++14

miscNodes: ../src/node/nodelogit.h ../src/node/noderelu.h ../src/node/nodevariable.h ../src/node/nodegaussian.h
	g++ -o miscNodes ../src/node/nodelogit.h ../src/node/noderelu.h ../src/node/nodevariable.h ../src/node/nodegaussian.h -I /usr/include/eigen3/ -std=c++14
