#include "testsHeader.h"

TEST(Random, SetSeed)
{
    int integers[3][10];
    float floats[3][10];
    double doubles[3][10];
    
    r.set_seed(42);
    
    for(int i = 0; i < 10; i++)
        integers[0][i] = r.rnd_int(i, 100);
    
    for(int i = 0; i < 10; i++)
        floats[0][i] = r.rnd_flt();
        
    for(int i = 0; i < 10; i++)
        doubles[0][i] = r.rnd_dbl();
        
        
    r.set_seed(10);
    
    for(int i = 0; i < 10; i++)
        integers[1][i] = r.rnd_int(i, 100);
    
    for(int i = 0; i < 10; i++)
        floats[1][i] = r.rnd_flt();
        
    for(int i = 0; i < 10; i++)
        doubles[1][i] = r.rnd_dbl();
        
    r.set_seed(98);
    
    for(int i = 0; i < 10; i++)
        integers[2][i] = r.rnd_int(i, 100);
    
    for(int i = 0; i < 10; i++)
        floats[2][i] = r.rnd_flt();
        
    for(int i = 0; i < 10; i++)
        doubles[2][i] = r.rnd_dbl();
        
    r.set_seed(42);
    
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(integers[0][i], r.rnd_int(i, 100));
        
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(floats[0][i], r.rnd_flt());
        
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(doubles[0][i], r.rnd_dbl());
    
    r.set_seed(10);
    
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(integers[1][i], r.rnd_int(i, 100));
        
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(floats[1][i], r.rnd_flt());
        
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(doubles[1][i], r.rnd_dbl());
    
    r.set_seed(98);
    
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(integers[2][i], r.rnd_int(i, 100));
        
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(floats[2][i], r.rnd_flt());
        
    for(int i = 0; i < 10; i++)
        ASSERT_EQ(doubles[2][i], r.rnd_dbl());
        
}
