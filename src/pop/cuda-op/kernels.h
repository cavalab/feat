/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef KERNELS 
#define KERNELS 
namespace FT{
    namespace Pop{
        namespace Op{
            // math operators
            void GPU_Add(float * x, size_t idx, size_t N, float W0, float W1);
            void GPU_Subtract(float * x, size_t idx, size_t N, float W0, float W1);
            void GPU_Multiply(float * x, size_t idx, size_t N, float W0, float W1);
            void GPU_Divide(float * x, size_t idx, size_t N, float W0, float W1);
            void GPU_Exp(float * x, size_t idx, size_t N, float W0);
            void GPU_Log(float * x, size_t idx, size_t N, float W0);
            void GPU_Sin(float * x, size_t idx, size_t N, float W0);
            void GPU_Cos(float * x, size_t idx, size_t N, float W0);
            void GPU_Sqrt(float * x, size_t idx, size_t N, float W0);
            void GPU_Square(float * x, size_t idx, size_t N, float W0);
            void GPU_Cube(float * x, size_t idx, size_t N, float W0);
            void GPU_Exponent(float * x, size_t idx, size_t N, float W0, float W1);
            void GPU_Logit(float * x, size_t idx, size_t N, float W0);
            void GPU_Step(float * x, size_t idx, size_t N);
            void GPU_Sign(float * x, size_t idx, size_t N);
            void GPU_Tanh(float * x, size_t idx, size_t N, float W0);
            void GPU_Gaussian(float * x, size_t idx, size_t N, float W0);
            void GPU_Gaussian2D(float * x, size_t idx,
                                float x1mean, float x1var,
                                float x2mean, float x2var,
                                float W0, float W1, size_t N);
            void GPU_Relu(float * x, size_t idx, size_t N, float W0);
            void GPU_Float(float * x, bool* y, size_t idxf, size_t idxb, size_t N);
            void GPU_Float(float * x, int* y, size_t idxf, size_t idxb, size_t N);

            // boolean operators
            void GPU_And(bool * x, size_t idx, size_t N);
            void GPU_Or(bool * x, size_t idx, size_t N);
            void GPU_Not(bool * x, size_t idx, size_t N);
            void GPU_Xor(bool * x, size_t idx, size_t N);
            void GPU_GEQ(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
            void GPU_LEQ(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
            void GPU_Equal(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
            void GPU_LessThan(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
            void GPU_GreaterThan(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
            void GPU_If(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
            void GPU_IfThenElse(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N);
            
            //learn
            void GPU_Split(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N, float threshold);
            void GPU_Split(int * xi, bool * xb, size_t idxi, size_t idxb, size_t N, float threshold);

            // leaves
            void GPU_Constant(float * dev_x, float value, size_t idx, size_t N);
            void GPU_Constant(bool * dev_x, bool value, size_t idx, size_t N);
            void GPU_Variable(float * dev_x, float * host_x, size_t idx, size_t N);
            void GPU_Variable(int * dev_x, int * host_x, size_t idx, size_t N);
            void GPU_Variable(bool * dev_x, bool * host_x, size_t idx, size_t N);
        }
    }
}
#endif
