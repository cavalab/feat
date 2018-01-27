# Fewtwo

**Fewtwo** is a feature engineering wrapper that learns new representations of raw data 
to improve classifier and regressor performance. The underlying methods are based on Pareto 
optimization and evolutionary computation to search the space of possible transformations.

Fewtwo is a completely different code base (and method) from [Few](https://lacava.github.io/few). 
The main differences are:

 - Each individual in Fewtwo's population is its own data representation, instead of one piece of an
   overall model
 - Fewtwo is pure c++
 - Fewtwo uses the [Shogun C++ ML toolbox](http://shogun.ml) instead of Scikit-learn


## Acknowledgments

This method is being developed to study the genetic causes of human disease in the [Epistasis Lab
at UPenn](http://epistasis.org). 

## License

GNU GPLv3
