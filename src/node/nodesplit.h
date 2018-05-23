/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LESSTHAN
#define NODE_LESSTHAN

#include "node.h"

namespace FT{
	class NodeSplit : public Node
    {
    	public:
    
            double threshold; 

    		NodeSplit()
       		{
    			name = "split";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}
            /// Uses a heuristic to set a splitting threshold.
            void set_threshold(ArrayXd& x, ArrayXd& y);

            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                ArrayXd x1 = stack.f.pop();
                if (!data.y.empty())
                    set_threshold(x1,data.y);

                stack.b.push(x1 < threshold);
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.fs.pop() + "<" + std::to_str(threshold) + ")");
            }
        protected:
            NodeSplit* clone_impl() const override { return new NodeSplit(*this); };  
            NodeSplit* rnd_clone_impl() const override { return new NodeSplit(); };  
    };
    
    void NodeSplit::set_threshold(ArrayXd& x, ArrayXd& y)
    {
        // for each unique value in x, calculate the reduction in the heuristic brought about by
        // splitting between that value and the next. 
        // set threshold according to the biggest reduction. 
        vector<double> s = x; //(x.data(),x.size());
        vector<size_t> idx(s.size());
        std::iota(idx.begin(),idx.end());
        s = unique(s);
        double score = 0;
        double best_score = 0;

        for (unsigned i =0; i<s.size(); ++i)
        {
            double val = (s.at(i) + s.at(i+1)) / 2;
            ArrayXi split_idx = (x < val).select(idx,-idx-1);
            // split data
            vector<double> d1, d2; 
            for (unsigned j=0; j< split_idx.size(); ++j)
            {
                if (split_idx(j) <0)
                    d2.push_back(y(-1-split_idx(j)));
                else
                    d1.push_back(y(split_idx(j)));
            }
            Map<VectorXd> map_d1(d1.data(), d1.size());  
            Map<VectorXd> map_d2(d2.data(), d2.size());  
            score = gain(map_d1, map_d2);
            if (score > best_score || i == 0)
            {
                threshold = val;
            }
        }

    }
   
    double NodeSplit::gain(const VectorXd& lsplit, const VectorXd& rsplit, 
            bool classification=true)
    {
        double lscore, rscore, score;
        if (classification)
        {
            lscore = gini_impurity_index(lsplit,lweight);
            rscore = gini_impurity_index(rsplit,rweight);
            score = lscore/double(lsplit.size()) + rscore/double(rsplit.size());
        }
        else
        {
            lscore = variance(lsplit.array());
            rscore = variance(rsplit.array());
            score = lscore + rscore; 
        }

	    return score;
    }
    double NodeSplit::gini_impurity_index(const VectorXd& classes)
    {
        double total_weight=classes.sum();
        double gini = classes.dot(classes);

        gini=1.0-(gini/(total_weight*total_weight));
        return gini;
    }

    double NodeSplit::least_squares_deviation(const VectorXd& feats)
    {

        return variance(feats.array());
        /* Map<VectorXd> map_weights(weights.vector, weights.size()); */
        /* Map<VectorXd> map_feats(feats.vector, weights.size()); */
        /* float64_t mean=map_weights.dot(map_feats); */
        /* total_weight=map_weights.sum(); */

        /* mean/=total_weight; */
        /* float64_t dev=0; */
        /* for (int32_t i=0;i<weights.vlen;i++) */
        /*     dev+=weights[i]*(feats[i]-mean)*(feats[i]-mean); */

        /* return dev/total_weight; */
    }
}	

#endif
