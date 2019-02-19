/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include "MyRandomForest.h"
#include "MyRandomCARTree.h"

using namespace shogun;

CMyRandomForest::CMyRandomForest()
: CBaggingMachine()
{	
	m_machine = new CMyRandomCARTree();
	m_weights=SGVector<float64_t>();

	SG_ADD(&m_weights,"m_weights","weights",MS_NOT_AVAILABLE)
}

CMyRandomForest::~CMyRandomForest()
{
}

void CMyRandomForest::set_weights(SGVector<float64_t> weights)
{
	m_weights=weights;
}

SGVector<float64_t> CMyRandomForest::get_weights() const
{
	return m_weights;
}

void CMyRandomForest::set_feature_types(SGVector<bool> ft)
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RandomCARTree\n")
	dynamic_cast<CMyRandomCARTree*>(m_machine)->set_feature_types(ft);
}

SGVector<bool> CMyRandomForest::get_feature_types() const
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RandomCARTree\n")
	return dynamic_cast<CMyRandomCARTree*>(m_machine)->get_feature_types();
}

EProblemType CMyRandomForest::get_machine_problem_type() const
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RandomCARTree\n")
	return dynamic_cast<CMyRandomCARTree*>(m_machine)->get_machine_problem_type();
}

void CMyRandomForest::set_machine_problem_type(EProblemType mode)
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RandomCARTree\n")
	dynamic_cast<CMyRandomCARTree*>(m_machine)->set_machine_problem_type(mode);
}

void CMyRandomForest::set_num_random_features(int32_t rand_featsize)
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RandomCARTree\n")
	REQUIRE(rand_featsize>0,"feature subset size should be greater than 0\n")

	dynamic_cast<CMyRandomCARTree*>(m_machine)->set_feature_subset_size(rand_featsize);
}

int32_t CMyRandomForest::get_num_random_features() const
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RandomCARTree\n")
	return dynamic_cast<CMyRandomCARTree*>(m_machine)->get_feature_subset_size();
}

void CMyRandomForest::set_machine_parameters(CMachine* m, SGVector<index_t> idx)
{
	REQUIRE(m,"Machine supplied is NULL\n")
	REQUIRE(m_machine,"Reference Machine is NULL\n")

	CMyRandomCARTree* tree=dynamic_cast<CMyRandomCARTree*>(m);

	SGVector<float64_t> weights(idx.vlen);

	if (m_weights.vlen==0)
	{
		weights.fill_vector(weights.vector,weights.vlen,1.0);
	}
	else
	{
		for (int32_t i=0;i<idx.vlen;i++)
			weights[i]=m_weights[idx[i]];
	}

	tree->set_weights(weights);
	tree->set_sorted_features(m_sorted_transposed_feats, m_sorted_indices);
	// equate the machine problem types - cloning does not do this
	tree->set_machine_problem_type(dynamic_cast<CMyRandomCARTree*>(m_machine)->get_machine_problem_type());
	
}

bool CMyRandomForest::train_machine(CFeatures* data)
{
	if (data)
	{
		SG_REF(data);
		SG_UNREF(m_features);
		m_features = data;
	}
	
	REQUIRE(m_features, "Training features not set!\n");
	
	dynamic_cast<CMyRandomCARTree*>(m_machine)->pre_sort_features(m_features, m_sorted_transposed_feats, m_sorted_indices);

	return CBaggingMachine::train_machine();
}

std::vector<double> CMyRandomForest::feature_importances()
{
     size_t num_features = get_feature_types().size();
     vector<double> importances(num_features, 0.0);    //set to zero for all attributes
     
     for (int32_t i = 0; i < m_num_bags; ++i)
     {  
         CMyRandomCARTree* m = dynamic_cast<CMyRandomCARTree*>(m_bags->get_element(i));
         
         vector<double> m_imp = m->feature_importances();
         
         for(size_t j = 0; j < num_features; j++)
            importances[j] += m_imp[j];
     }
     
     for(size_t i = 0; i < num_features; i++)
            importances[i] += m_num_bags;
            
     return importances;
}

void CMyRandomForest::set_probabilities(CLabels* labels, CFeatures* data)
{
    SGMatrix<float64_t> output = apply_outputs_without_combination(data);

    CMeanRule* mean_rule = new CMeanRule();

    SGVector<float64_t> probabilities = mean_rule->combine(output);

    labels->set_values(probabilities);

    SG_UNREF(mean_rule);
}  
