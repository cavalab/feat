#include "testsHeader.h"
#include "cudaTestUtils.h"

#define NEAR_ZERO 0.0001

State evaluateNodes(NodeVector &nodes, MatrixXf &X, string testNode)
{

    cout << "Running for " << testNode << endl;
     
    State state;
    
    VectorXf Y(6); 
    Y << 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z1;
    
    Data data(X, Y, z1);
    
    #ifndef USE_CUDA
    
    for (const auto& n : nodes)   
        n->evaluate(data, state);
    
    #else
    
    std::map<char, size_t> state_size = get_max_state_size(nodes);
    
    choose_gpu();
    
    state.allocate(state_size,data.X.cols());
    
    for (const auto& n : nodes)   
    {
        n->evaluate(data, state);
        state.update_idx(n->otype, n->arity);
    }	
    
    state.copy_to_host();
    
    #endif
    
    return state;
    
    
}

void compareStates(State output, State expected, char otype)
{
    #ifndef USE_CUDA
    
    if(otype == 'f')
    {
        int index = output.size<float>()-1;
        
        /* cout << "Output is\n" << output.get<float>()[index] << endl; */
        /* cout << "Expected is\n" << expected.get<float>()[index] << endl; */
        /* cout << "Difference is\n" << abs(output.get<float>()[index] - expected.get<float>()[index]) << endl; */
        
        ASSERT_TRUE((abs(output.get<float>()[index] - expected.get<float>()[index]) < NEAR_ZERO).all());
        ASSERT_FALSE((isinf(output.get<float>()[index])).any());
        ASSERT_FALSE((isnan(abs(output.get<float>()[index])).any()));
    }
    
    if(otype == 'b')
    {
        int index = output.size<bool>()-1;
        
        ArrayXb exp = expected.get<float>()[index].template cast<bool>();
        ArrayXb out = output.get<bool>()[index];
        
        /* cout << "Output is\n" << out << endl; */
        /* cout << "Expected is\n" << exp << endl; */
        //cout << "Difference is\n" << abs(output.get<float>()[index] - expected.get<float>()[index]) << endl;
       
        ASSERT_TRUE((abs(out - exp) < NEAR_ZERO).all());
        ASSERT_FALSE((isinf(out)).any());
        ASSERT_FALSE((isnan(abs(out)).any())); 
    }
    
    #else
    
    if(otype == 'f')
    {
        /* cout << "Output is\n" << output.f << endl; */
        /* cout << "Expected is\n" << expected.f << endl; */
        /* cout << "Difference is\n" << abs(output.f - expected.f) << endl; */
        
        ASSERT_TRUE((abs(output.f - expected.f) < NEAR_ZERO).all());
        ASSERT_FALSE((isinf(output.f)).any());
        ASSERT_FALSE((isnan(abs(output.f)).any()));
    }
    
    if(otype == 'b')
    {        
        /* cout << "Output is\n" << output.b << endl; */
        /* cout << "Expected is\n" << expected.f << endl; */
        
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> exp = expected.f.template cast<bool>();
        
        ASSERT_TRUE((abs(output.b - exp) < NEAR_ZERO).all());
        ASSERT_FALSE((isinf(output.b)).any());
        ASSERT_FALSE((isnan(abs(output.b)).any())); 
    }
    #endif
}

template <class type>
State createExpectedState(vector<type> expectedValues)
{
    State state;
    
    int size = expectedValues.size();
    
    #ifndef USE_CUDA
    
    Map<Eigen::Array<type,Eigen::Dynamic,1> > arr(expectedValues.data(), size);
    state.push<type>(arr);
    
    #else
    
    state.f.resize(1, size);
    
    for(int x = 0; x < expectedValues.size(); x++)
        state.f(0, x) = expectedValues[x];
    
    #endif
    
    return state;

}

TEST(NodeTest, Evaluate)
{    
    initialize_cuda(); 

	MatrixXf X1(2,3); 
    X1 << 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0;   
   
    NodeVector nodes;
    
    std::unique_ptr<Node> f1 = std::unique_ptr<Node>(new NodeVariable<float>(0));
    std::unique_ptr<Node> f2 = std::unique_ptr<Node>(new NodeVariable<float>(1));
    
    std::unique_ptr<Node> gauss2d = std::unique_ptr<Node>(new Node2dGaussian({1.0, 1.0}));
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(gauss2d->clone());

    compareStates(evaluateNodes(nodes, X1, "2dguass"), createExpectedState<float>({0.105399, 1.0, 0.105399}), 'f');
    
    std::unique_ptr<Node> addObj = std::unique_ptr<Node>(new NodeAdd({1.0, 1.0}));
    
    nodes.clear();

    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(addObj->clone());

    compareStates(evaluateNodes(nodes, X1, "add"), createExpectedState<float>({5.0, 7.0, 9.0}), 'f');
    
    std::unique_ptr<Node> subObj = std::unique_ptr<Node>(new NodeSubtract({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(subObj->clone());

    compareStates(evaluateNodes(nodes, X1, "subtract"), createExpectedState<float>({3.0, 3.0, 3.0}), 'f');
    
    std::unique_ptr<Node> mulObj = std::unique_ptr<Node>(new NodeMultiply({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(mulObj->clone());

    compareStates(evaluateNodes(nodes, X1, "multiply"), createExpectedState<float>({4.0, 10.0, 18.0}), 'f');
    
    std::unique_ptr<Node> divObj = std::unique_ptr<Node>(new NodeDivide({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(divObj->clone());

    compareStates(evaluateNodes(nodes, X1, "divide"), createExpectedState<float>({4.0, 2.5, 2.0}), 'f');
    
    std::unique_ptr<Node> expObj = std::unique_ptr<Node>(new NodeExponent({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(expObj->clone());

    compareStates(evaluateNodes(nodes, X1, "exponent"), createExpectedState<float>({4.0, 25.0, 216.0}), 'f');
    
    MatrixXf X2(1,4); 
    X2 << 0.0, 1.0, 2.0, 3.0;
    
    std::unique_ptr<Node> cosObj = std::unique_ptr<Node>(new NodeCos({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(cosObj->clone());

    compareStates(evaluateNodes(nodes, X2, "cos"), createExpectedState<float>({1, 0.540302, -0.416147, -0.989992}), 'f');
    
    std::unique_ptr<Node> cubeObj = std::unique_ptr<Node>(new NodeCube({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(cubeObj->clone());

    compareStates(evaluateNodes(nodes, X2, "cube"), createExpectedState<float>({0.0, 1, 8, 27}), 'f');
    
    std::unique_ptr<Node> exptObj = std::unique_ptr<Node>(new NodeExponential({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(exptObj->clone());

    compareStates(evaluateNodes(nodes, X2, "exponential"), createExpectedState<float>({1, 2.71828, 7.38906, 20.0855}), 'f');
    
    std::unique_ptr<Node> gaussObj = std::unique_ptr<Node>(new NodeGaussian({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(gaussObj->clone());

    compareStates(evaluateNodes(nodes, X2, "gaussian"), createExpectedState<float>({0.367879, 1, 0.367879, 0.0183156}), 'f');
    
    std::unique_ptr<Node> logObj = std::unique_ptr<Node>(new NodeLog({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(logObj->clone());

    compareStates(evaluateNodes(nodes, X2, "log"), createExpectedState<float>({MIN_FLT, 0, 0.693147, 1.09861}), 'f');
    
    std::unique_ptr<Node> logitObj = std::unique_ptr<Node>(new NodeLogit({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(logitObj->clone());

    compareStates(evaluateNodes(nodes, X2, "logit"), createExpectedState<float>({0.5, 0.731059, 0.880797, 0.952574}), 'f');
    
    std::unique_ptr<Node> reluObj = std::unique_ptr<Node>(new NodeRelu({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(reluObj->clone());

    compareStates(evaluateNodes(nodes, X2, "relu"), createExpectedState<float>({0.01, 1, 2, 3}), 'f');
    
    std::unique_ptr<Node> signObj = std::unique_ptr<Node>(new NodeSign());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(signObj->clone());

    compareStates(evaluateNodes(nodes, X2, "sign"), createExpectedState<float>({0, 1, 1, 1}), 'f');
    
    std::unique_ptr<Node> sinObj = std::unique_ptr<Node>(new NodeSin({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(sinObj->clone());

    compareStates(evaluateNodes(nodes, X2, "sin"), createExpectedState<float>({0, 0.841471, 0.909297, 0.14112}), 'f');
    
    std::unique_ptr<Node> sqrtObj = std::unique_ptr<Node>(new NodeSqrt({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(sqrtObj->clone());

    compareStates(evaluateNodes(nodes, X2, "sqrt"), createExpectedState<float>({0, 1, 1.41421, 1.73205}), 'f');
    
    std::unique_ptr<Node> squareObj = std::unique_ptr<Node>(new NodeSquare({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(squareObj->clone());

    compareStates(evaluateNodes(nodes, X2, "square"), createExpectedState<float>({0, 1, 4, 9}), 'f');
    
    std::unique_ptr<Node> stepObj = std::unique_ptr<Node>(new NodeStep());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(stepObj->clone());

    compareStates(evaluateNodes(nodes, X2, "step"), createExpectedState<float>({0, 1, 1, 1}), 'f');
    
    std::unique_ptr<Node> tanObj = std::unique_ptr<Node>(new NodeTanh({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(tanObj->clone());

    compareStates(evaluateNodes(nodes, X2, "tan"), createExpectedState<float>({0, 0.761594, 0.964028, 0.995055}), 'f');
    
    std::unique_ptr<Node> split_fObj = std::unique_ptr<Node>(new NodeSplit<float>());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(split_fObj->clone());

    compareStates(evaluateNodes(nodes, X2, "split_float"), createExpectedState<float>({0, 0, 0, 0}), 'b');
    
    std::unique_ptr<Node> i1 = std::unique_ptr<Node>(new NodeVariable<int>(0, 'c'));
    std::unique_ptr<Node> split_iObj = std::unique_ptr<Node>(new NodeSplit<int>());
    
    nodes.clear();
    
    nodes.push_back(i1->clone());
    nodes.push_back(split_iObj->clone());

    compareStates(evaluateNodes(nodes, X2, "split_int"), createExpectedState<float>({1, 0, 0, 0}), 'b');
    
    MatrixXf X3(2,3); 
    X3 << 0.0, 1.0, 1.0,
          0.0, 1.0, 0.0;
          
    std::unique_ptr<Node> b1 = std::unique_ptr<Node>(new NodeVariable<bool>(0, 'b'));
    std::unique_ptr<Node> b2 = std::unique_ptr<Node>(new NodeVariable<bool>(1, 'b'));
          
    std::unique_ptr<Node> andObj = std::unique_ptr<Node>(new NodeAnd());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(andObj->clone());

    compareStates(evaluateNodes(nodes, X3, "and"), createExpectedState<float>({0, 1, 0}), 'b');
    
    std::unique_ptr<Node> orObj = std::unique_ptr<Node>(new NodeOr());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(orObj->clone());

    compareStates(evaluateNodes(nodes, X3, "or"), createExpectedState<float>({0, 1, 1}), 'b');
    
    std::unique_ptr<Node> xorObj = std::unique_ptr<Node>(new NodeXor());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(xorObj->clone());

    compareStates(evaluateNodes(nodes, X3, "xor"), createExpectedState<float>({0, 0, 1}), 'b');
    
    MatrixXf X4(2,3); 
    X4 << 1.0, 2.0, 3.0,
          1.0, 1.0, 4.0;
    
    std::unique_ptr<Node> eqObj = std::unique_ptr<Node>(new NodeEqual());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(eqObj->clone());

    compareStates(evaluateNodes(nodes, X4, "equal"), createExpectedState<float>({1, 0, 0}), 'b');
    
    std::unique_ptr<Node> geqObj = std::unique_ptr<Node>(new NodeGEQ());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(geqObj->clone());

    compareStates(evaluateNodes(nodes, X4, "GEQ"), createExpectedState<float>({1, 0, 1}), 'b');
    
    std::unique_ptr<Node> gtObj = std::unique_ptr<Node>(new NodeGreaterThan());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(gtObj->clone());

    compareStates(evaluateNodes(nodes, X4, "GreaterThan"), createExpectedState<float>({0, 0, 1}), 'b');
    
    std::unique_ptr<Node> leqObj = std::unique_ptr<Node>(new NodeLEQ());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(leqObj->clone());

    compareStates(evaluateNodes(nodes, X4, "LEQ"), createExpectedState<float>({1, 1, 0}), 'b');
    
    std::unique_ptr<Node> ltObj = std::unique_ptr<Node>(new NodeLessThan());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(ltObj->clone());

    compareStates(evaluateNodes(nodes, X4, "LessThan"), createExpectedState<float>({0, 1, 0}), 'b');
    
    MatrixXf X5(2,3); 
    X5 << 1.0, 2.0, 3.0,
          1.0, 0.0, 1.0;
          
    std::unique_ptr<Node> ifObj = std::unique_ptr<Node>(new NodeIf());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(b2->clone());
    nodes.push_back(ifObj->clone());

    compareStates(evaluateNodes(nodes, X5, "If"), createExpectedState<float>({1, 0, 3}), 'f');
    
    MatrixXf X6(3,3); 
    X6 << 1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          0.0, 1.0, 0.0;
          
    std::unique_ptr<Node> b3 = std::unique_ptr<Node>(new NodeVariable<bool>(2, 'b'));
          
    std::unique_ptr<Node> iteObj = std::unique_ptr<Node>(new NodeIfThenElse());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(b3->clone());
    nodes.push_back(iteObj->clone());

    compareStates(evaluateNodes(nodes, X6, "IfThenElse"), createExpectedState<float>({1, 5, 3}), 'f');
    
    MatrixXf X7(1,2); 
    X7 << 1, 0;
    
    std::unique_ptr<Node> notObj = std::unique_ptr<Node>(new NodeNot());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(notObj->clone());

    compareStates(evaluateNodes(nodes, X7, "Not"), createExpectedState<float>({0, 1}), 'b');
    
    std::unique_ptr<Node> float_bObj = std::unique_ptr<Node>(new NodeFloat<bool>());
    
    nodes.clear();
    
    nodes.push_back(b1->clone());
    nodes.push_back(float_bObj->clone());

    compareStates(evaluateNodes(nodes, X7, "float_bool"), createExpectedState<float>({1.0, 0.0}), 'f');
    
    std::unique_ptr<Node> float_iObj = std::unique_ptr<Node>(new NodeFloat<int>());
    
    nodes.clear();
    
    nodes.push_back(i1->clone());
    nodes.push_back(float_iObj->clone());

    compareStates(evaluateNodes(nodes, X2, "float_int"), createExpectedState<float>({0.0, 1.0, 2.0, 3.0}), 'f');
    
    //TODO NodeVariable, NodeConstant(both types) maybe not required....var tested in all other tests
    // so is constant
}

