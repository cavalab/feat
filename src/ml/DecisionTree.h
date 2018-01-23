/* FEWTWO
copyright 2017 William La Cava 
license: GNU/GPL v3
*/

#ifndef DecisionTree_H
#define DecisionTree_H

//external includes

#include <shogun/multiclass/tree/CARTree.h>
namespace FT{
    namespace ml {
        
        /*! 
         * @class DecisionTree
         * @brief a wrapper class for CART trees that gets feature importance scores.
         */
        class DecisionTree : sh::CC45ClassifierTree
        {
            // method to return importance scores. 
            vector<double> feature_importance()
            {
                /*! Importance is defined as the sum across all splits in the tree of the
                 * information criterion brought about by each feature. 
                 */

                 vector<double> importances;

                 auto node = m_root;
                 // make this recursive! 
                 //
                 while (node->num_leaves > 0 )
                 {
                     importances[node->attribute_id] += impurity(node) 
                                                        - impurity(node->m_children[0])
                                                        - impurity(node->m_children[1])
                     // traverse all nodes
                     ++node;
                 }
                 // while node != end node
                 //     if not a leaf:
                 //     importances[node.feature ] += node.impurity - left.impurity - right.impurity
                 // normalize weights
                 //
                 // shogun names:
                 //     node.feature = attribute_id (int32_t)
                 //     node.impurity = total_weight of training samples passing thru this node
                 //     float64_t weight_minus_node : total weight of misclassified samples in node
                 //     weight_minus_branch: total weight of misclassified samples in subtree
            }
        }
        double impurity ( CtreeMachineNode * n )
        {
            /*! computes the impurity of a node. 
             */
            return n->total_weight * (1 - weight_minus_node);
        }
    }

#endif
