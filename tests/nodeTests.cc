#include "testsHeader.h"
#include "cudaTestUtils.h"

#define NEAR_ZERO 0.0001

State evaluateNodes(NodeVector &nodes, MatrixXf &X, string testNode, VectorXf Y=VectorXf(),
        bool classification=false)
{

    cout << "Running for " << testNode << endl;
     
    State state;
   
    if (Y.size() == 0)
    {
        cout << "Y is empty, setting...\n";
        Y.resize(4);
        Y << 3.0, 4.0, 5.0, 6.0; //, 7.0, 8.0;
    }

    cout << "set Y, continuing...\n";
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z1;
    
    Data data(X, Y, z1);
    data.classification= classification;    
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
        
        cout << "Output is\n" << out.transpose() << endl;
        cout << "Expected is\n" << exp.transpose() << endl;
        /* cout << "Difference is\n" << abs(output.get<float>()[index] - expected.get<float>()[index]) << endl; */
       
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

    compareStates(evaluateNodes(nodes, X1, "2dguass"), 
            createExpectedState<float>({0.105399, 1.0, 0.105399}), 'f');
    
    std::unique_ptr<Node> addObj = std::unique_ptr<Node>(new NodeAdd({1.0, 1.0}));
    
    nodes.clear();

    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(addObj->clone());

    compareStates(evaluateNodes(nodes, X1, "add"), 
            createExpectedState<float>({5.0, 7.0, 9.0}), 'f');
    
    std::unique_ptr<Node> subObj = std::unique_ptr<Node>(new NodeSubtract({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(subObj->clone());

    compareStates(evaluateNodes(nodes, X1, "subtract"), 
            createExpectedState<float>({3.0, 3.0, 3.0}), 'f');
    
    std::unique_ptr<Node> mulObj = std::unique_ptr<Node>(new NodeMultiply({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(mulObj->clone());

    compareStates(evaluateNodes(nodes, X1, "multiply"), 
            createExpectedState<float>({4.0, 10.0, 18.0}), 'f');
    
    std::unique_ptr<Node> divObj = std::unique_ptr<Node>(new NodeDivide({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(divObj->clone());

    compareStates(evaluateNodes(nodes, X1, "divide"), 
            createExpectedState<float>({4.0, 2.5, 2.0}), 'f');
    
    std::unique_ptr<Node> expObj = std::unique_ptr<Node>(new NodeExponent({1.0, 1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(f2->clone());
    nodes.push_back(expObj->clone());

    compareStates(evaluateNodes(nodes, X1, "exponent"), 
            createExpectedState<float>({4.0, 25.0, 216.0}), 'f');
    
    MatrixXf X2(1,4); 
    X2 << 0.0, 1.0, 2.0, 3.0;
    
    std::unique_ptr<Node> cosObj = std::unique_ptr<Node>(new NodeCos({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(cosObj->clone());

    compareStates(evaluateNodes(nodes, X2, "cos"), 
            createExpectedState<float>({1, 0.540302, -0.416147, -0.989992}), 'f');
    
    std::unique_ptr<Node> cubeObj = std::unique_ptr<Node>(new NodeCube({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(cubeObj->clone());

    compareStates(evaluateNodes(nodes, X2, "cube"), 
            createExpectedState<float>({0.0, 1, 8, 27}), 'f');
    
    std::unique_ptr<Node> exptObj = std::unique_ptr<Node>(new NodeExponential({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(exptObj->clone());

    compareStates(evaluateNodes(nodes, X2, "exponential"), 
            createExpectedState<float>({1, 2.71828, 7.38906, 20.0855}), 'f');
    
    std::unique_ptr<Node> gaussObj = std::unique_ptr<Node>(new NodeGaussian({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(gaussObj->clone());

    compareStates(evaluateNodes(nodes, X2, "gaussian"), 
            createExpectedState<float>({0.367879, 1, 0.367879, 0.0183156}), 'f');
    
    std::unique_ptr<Node> logObj = std::unique_ptr<Node>(new NodeLog({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(logObj->clone());

    compareStates(evaluateNodes(nodes, X2, "log"), 
            createExpectedState<float>({MIN_FLT, 0, 0.693147, 1.09861}), 'f');
    
    std::unique_ptr<Node> logitObj = std::unique_ptr<Node>(new NodeLogit({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(logitObj->clone());

    compareStates(evaluateNodes(nodes, X2, "logit"), 
            createExpectedState<float>({0.5, 0.731059, 0.880797, 0.952574}), 'f');
    
    std::unique_ptr<Node> reluObj = std::unique_ptr<Node>(new NodeRelu({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(reluObj->clone());

    compareStates(evaluateNodes(nodes, X2, "relu"), 
            createExpectedState<float>({0.01, 1, 2, 3}), 'f');
    
    std::unique_ptr<Node> signObj = std::unique_ptr<Node>(new NodeSign());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(signObj->clone());

    compareStates(evaluateNodes(nodes, X2, "sign"), 
            createExpectedState<float>({0, 1, 1, 1}), 'f');
    
    std::unique_ptr<Node> sinObj = std::unique_ptr<Node>(new NodeSin({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(sinObj->clone());

    compareStates(evaluateNodes(nodes, X2, "sin"), 
            createExpectedState<float>({0, 0.841471, 0.909297, 0.14112}), 'f');
    
    std::unique_ptr<Node> sqrtObj = std::unique_ptr<Node>(new NodeSqrt({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(sqrtObj->clone());

    compareStates(evaluateNodes(nodes, X2, "sqrt"), 
            createExpectedState<float>({0, 1, 1.41421, 1.73205}), 'f');
    
    std::unique_ptr<Node> squareObj = std::unique_ptr<Node>(new NodeSquare({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(squareObj->clone());

    compareStates(evaluateNodes(nodes, X2, "square"), 
            createExpectedState<float>({0, 1, 4, 9}), 'f');
    
    std::unique_ptr<Node> stepObj = std::unique_ptr<Node>(new NodeStep());
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(stepObj->clone());

    compareStates(evaluateNodes(nodes, X2, "step"), 
            createExpectedState<float>({0, 1, 1, 1}), 'f');
    
    std::unique_ptr<Node> tanObj = std::unique_ptr<Node>(new NodeTanh({1.0}));
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(tanObj->clone());

    compareStates(evaluateNodes(nodes, X2, "tan"), 
            createExpectedState<float>({0, 0.761594, 0.964028, 0.995055}), 'f');
    
    std::unique_ptr<Node> split_fObj = std::unique_ptr<Node>(new NodeSplit<float>());
    dynamic_cast<NodeTrain*>(split_fObj.get())->train = true;
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(split_fObj->clone());

    compareStates(evaluateNodes(nodes, X2, "split_float"), 
            createExpectedState<float>({1, 0, 0, 0}), 'b');
    
    std::unique_ptr<Node> i1 = std::unique_ptr<Node>(new NodeVariable<int>(0, 'c'));
    std::unique_ptr<Node> split_iObj = std::unique_ptr<Node>(new NodeSplit<int>());
    dynamic_cast<NodeTrain*>(split_fObj.get())->train = true;
    
    nodes.clear();
    
    nodes.push_back(i1->clone());
    nodes.push_back(split_iObj->clone());

    compareStates(evaluateNodes(nodes, X2, "split_int"), 
            createExpectedState<float>({1, 0, 0, 0}), 'b');
    
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

TEST(NodeTest, Split)
{    
    initialize_cuda(); 

    NodeVector nodes;
    
    std::unique_ptr<Node> f1 = std::unique_ptr<Node>(new NodeVariable<float>(0));

    std::unique_ptr<Node> split_fObj = std::unique_ptr<Node>(new NodeSplit<float>());
    dynamic_cast<NodeTrain*>(split_fObj.get())->train = true;
    
    nodes.clear();
    
    nodes.push_back(f1->clone());
    nodes.push_back(split_fObj->clone());

    // here is a hypothetical classification dataset with one feature with an optimal split of 
    // 29.849
    MatrixXf X(1,100);
    X << 29.080639469540202, 16.557798939045448, 34.934921462098444, 23.010654099228017, 
      22.422935475966717, 9.54588870489429, 4.754518078110699, 20.532653527940646, 
      17.117988486068068, 25.462676368094016, 21.682785565864997, 17.78129128408562, 
      26.488879356628765, 20.219571566281587, 10.923165784606965, 24.8414089559337, 
      30.617282935962884, 19.104462591592522, 26.19854837504999, 15.961391160147198, 
      40.20684669014179, 37.28789791394121, 18.229468870072786, 28.484572869133366, 
      31.592081655544092, 23.065088933243626, 38.61726630405504, 8.359144362613742,
      -0.7861456777433773, 17.088626469266877, 21.746260521639876, 17.61531719723882,
      42.334531193323095, 22.264863520020725, 28.25245999495475, 10.480031358059712, 
      20.864226469335563, 7.287166677740897, 11.206714697371996, 11.394732174505487, 
      16.506462542851963, 5.339221302558908, 31.744833338696772, 35.435569827378956, 
      27.757820591759337, 3.8360380481960235, 7.927792983587809, 17.828827374985078, 
      12.071325534483513, 24.35491537455122, 32.2219759400991, 21.100943542607656, 
      23.402472006795303, 19.40789471067006, 24.0490944831797, 27.82406324374343, 
      21.464668131323585, 26.692274915533154, 31.05031309778429, 53.149351801641714, 
      38.22778624389851, -4.330646678766108, 9.635115532745765, 10.314298478893175, 
      12.557291901239674, 13.914966880186599, 18.715438175258747, 25.31061649317862, 
      31.211181043236557, 4.068901674253635, 38.449639714117055, 19.24269903155262, 
      26.378902528958047, 25.815719544339938, 26.870782920413276, 24.289191844307414, 
      19.501413925272157, 31.012772897265357, 33.13972084460402, 0.09623450824753021, 
      17.928304452037093, 25.88016141806907, 22.012296345122557, 20.345015537140164, 
      8.1713253739893, 16.80642202671364, 0.611395743720621, 40.18480923430135, 
      22.79644205885677, 15.838105413509354, 9.039601330145409, 28.953620678980556, 
      15.324976877868117, 17.978429436920813, 28.635676101384426, 10.117846051899983, 
      19.36717740754287, 26.948031573512907, -6.48036628890862, 19.407803623483012; 
    VectorXf y(100);
    y << 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 
      1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 
      1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 
      1, 1, 1, 1, 1, 1, 1, 1 ;
    /* X = Map<MatrixXf>(values.data(), values.size()/(rows-1), rows-1); */
    /* cout << "X: " << X << "\n"; */
    /* cout << "X: " << X << "\n"; */
    /* cout << "y: " << y.transpose() << "\n"; */
   
    vector<float> expected_y(100);
    for (int i = 0; i<expected_y.size(); ++i)
        expected_y[i] = float(X(i) < 29.849);
    compareStates(evaluateNodes(nodes, X, "split_float", y, true), 
                  createExpectedState<float>(expected_y), 'b');
}
