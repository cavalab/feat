# FEW-two

**Few-two** is a feature engineering wrapper that wraps around the Shogun C++ machine learning 
toolkit and interfaces with scikit-learn. Its purpose is to learn new representations of raw data 
to improve classifier and regressor performance. The underlying methods are based on Pareto 
optimization and evolutionary computation to search the space of possible transformations.

FEW-two is a completely different code base from [Few](https://lacava.github.io/few). The main
differences are:

 - Each individual in FEW-two is its own ML model + data representation, instead of one piece of an
   overall model
 - FEW-two is pure c++
 - FEw-two uses the Shogun C++ ML toolbox instead of Scikit-learn


## Acknowledgments

This method is being developed to study the genetic causes of human disease in the [Epistasis Lab
at UPenn](http://epistasis.org). 

## License

GNU GPLv3
