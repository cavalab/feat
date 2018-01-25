/* FEWTWO
copyright 2017 William La Cava 
license: GNU/GPL v3
*/

#ifndef DecisionTree_H
#define DecisionTree_H

//external includes
#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/CARTree.h>

// stuff being used
using std::cout; 
namespace sh = shogun;

namespace FT{
    namespace ml {
        
        /*! 
         * @class DecisionTree
         * @brief a wrapper class for CART trees that gets feature importance scores.
         */
        class DecisionTree : public sh::CCARTree
        {
        public:
            typedef sh::CBinaryTreeMachineNode<sh::CARTreeNodeData> bnode_t;
            // method to return importance scores. 
            vector<double> feature_importances()
            {
                /*! Importance is defined as the sum across all splits in the tree of the
                 * information criterion brought about by each feature. 
                 */
                 // need to get feature sizes
                 sh::SGVector<bool> dt = get_feature_types();
                 vector<double> importances(dt.size(),0.0);    //set to zero for all attributes

                 bnode_t* node = dynamic_cast<bnode_t*>(m_root);
                 get_importance(node, importances); 
                 
                 return importances;
            }

            void get_importance(bnode_t* node, vector<double> importances)
            {

                 if (node->data.num_leaves!=1)
                 {              
                     bnode_t* left=node->left();
                     bnode_t* right=node->right();
                    
                     importances[node->data.attribute_id] += impurity(node) - impurity(left)
                                                            - impurity(right); 
                     get_importance(left,importances);
                     get_importance(right,importances);

                     SG_UNREF(left);
                     SG_UNREF(right);
                 }
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
            
            /// compute the impurity of a node.  
            double impurity ( bnode_t * n )
            {
                //n->data.print_data(n->data);
                double p0 = (1-n->data.weight_minus_node)/ n->data.total_weight;
                double p1 = n->data.weight_minus_node / n->data.total_weight;
                cout << "node num_leaves: " << n->data.num_leaves << "\n";
                cout << "node label: " << n->data.node_label << "\n";
                cout << "node p0: " << p0 << "\n";
                cout << "node p1: " << p1 << "\n";
                cout << "impurity: " << 1 - (pow(p0,2)+pow(p1,2)) << "\n---\n";
                return 1 - (pow(p0,2)+pow(p1,2));
            }
        };
    }
}
#endif
