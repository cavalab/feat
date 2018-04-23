all: autoBP

sample: ../src/node/node.h ../src/node/nodeDx.h trigNodes arithNodes powerNodes eNodes miscNodes
	g++ -o sample ../src/node/node.h ../src/node/nodeDx.h ../src/node/nodecos.h

autoBP: autoDiffTester.cc ../src/auto_backprop.h ../src/testMetrics.h nodes
	g++ -o autoBP autoDiffTester.cc ../src/auto_backprop.h ../src/testMetrics.h

nodes: ../src/node/node.h ../src/node/nodeDx.h trigNodes arithNodes powerNodes eNodes miscNodes
	g++ -o nodes ../src/node/node.h ../src/node/nodeDx.h

trigNodes: ../src/node/nodecos.h ../src/node/nodesin.h ../src/node/nodetanh.h
	g++ -o trigNodes ../src/node/nodecos.h ../src/node/nodesin.h ../src/node/nodetanh.h

arithNodes: ../src/node/nodeadd.h ../src/node/nodesubtract.h ../src/node/nodemultiply.h ../src/node/nodedivide.h
	g++ -o arithNodes ../src/node/nodeadd.h ../src/node/nodesubtract.h ../src/node/nodemultiply.h ../src/node/nodedivide.h

powerNodes: ../src/node/nodecube.h ../src/node/nodeexponent.h ../src/node/nodesqrt.h ../src/node/nodesquare.h
	g++ -o powerNodes ../src/node/nodecube.h ../src/node/nodeexponent.h ../src/node/nodesqrt.h ../src/node/nodesquare.h

eNodes: ../src/node/nodeexponential.h ../src/node/nodelog.h
	g++ -o eNodes ../src/node/nodeexponential.h ../src/node/nodelog.h

miscNodes: ../src/node/nodelogit.h ../src/node/noderelu.h ../src/node/nodevariable.h ../src/node/nodegaussian.h ../src/node/node2dgaussian.h
	g++ -o miscNodes ../src/node/nodelogit.h ../src/node/noderelu.h ../src/node/nodevariable.h ../src/node/nodegaussian.h ../src/node/node2dgaussian.h