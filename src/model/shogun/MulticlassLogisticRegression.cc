/* This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include "MulticlassLogisticRegression.h"

#include<iostream>  

namespace shogun
{
		
		const char* CMulticlassLogisticRegression::get_name() const
		{
			return "MulticlassLogisticRegression";
		}

		void CMulticlassLogisticRegression::set_z(float64_t z)
		{
			ASSERT(z>0)
			m_z = z;
		}
		
		inline float64_t CMulticlassLogisticRegression::get_z() const { return m_z; }

		void CMulticlassLogisticRegression::set_epsilon(float64_t epsilon)
		{
			ASSERT(epsilon>0)
			m_epsilon = epsilon;
		}
		
		inline float64_t CMulticlassLogisticRegression::get_epsilon() const { return m_epsilon; }

		void CMulticlassLogisticRegression::set_max_iter(int32_t max_iter)
		{
			ASSERT(max_iter>0)
			m_max_iter = max_iter;
		}
		
		inline int32_t CMulticlassLogisticRegression::get_max_iter() const { return m_max_iter; }
       		
    using namespace shogun;

    CMulticlassLogisticRegression::CMulticlassLogisticRegression() :
	    CLinearMulticlassMachine()
    {
	    init_defaults();
    }

    CMulticlassLogisticRegression::CMulticlassLogisticRegression(float64_t z, CDotFeatures* feats, CLabels* labs) :
	    CLinearMulticlassMachine(new CMulticlassOneVsRestStrategy(),feats,NULL,labs)
    {
	    init_defaults();
	    set_z(z);
    }

    void CMulticlassLogisticRegression::init_defaults()
    {
	    set_z(0.1);
	    set_epsilon(1e-2);
	    set_max_iter(10000);
    }

    void CMulticlassLogisticRegression::register_parameters()
    {
	    SG_ADD(&m_z, "m_z", "regularization constant",MS_AVAILABLE);
	    SG_ADD(&m_epsilon, "m_epsilon", "tolerance epsilon",MS_NOT_AVAILABLE);
	    SG_ADD(&m_max_iter, "m_max_iter", "max number of iterations",MS_NOT_AVAILABLE);
    }

    CMulticlassLogisticRegression::~CMulticlassLogisticRegression()
    {

		for (int i = 0; i < m_machines->get_num_elements(); ++i)
		{
			auto ptr = m_machines->get_element(i);
			while (ptr->ref_count() > 2)
			{
				SG_UNREF(ptr);
			}
			SG_UNREF(ptr);
		}
		m_machines->reset_array();

		/** machine */
	    if (m_features)
	    {
		    delete m_features;
		    m_features = NULL;
	    }
	    if (m_machines)
	    {
		    delete m_machines;
		    m_machines = NULL;
	    }
	    if (m_machine)
	    {
		    delete m_machine;
		    m_machine = NULL;
	    }
	    if (m_multiclass_strategy)
	    {
		    delete m_multiclass_strategy;
		    m_multiclass_strategy = NULL;
	    }
    }


    vector<SGVector<float64_t>> CMulticlassLogisticRegression::get_w()
    {
        vector<SGVector<float64_t>> weights_vector;
        
        int n_machines = get_num_machines();
        
        for (int32_t i=0; i<n_machines; i++)
	    {
		    CLinearMachine* machine = (CLinearMachine*)m_machines->get_element(i);
		    weights_vector.push_back(machine->get_w());
			SG_UNREF(machine);
	    }
	
        return weights_vector;	
    }

    vector<float64_t> CMulticlassLogisticRegression::get_bias()
    {
        vector<float64_t> bias_vector;
        
        int n_machines = get_num_machines();
        
        for (int32_t i=0; i<n_machines; i++)
	    {
		    CLinearMachine* machine = (CLinearMachine*)m_machines->get_element(i);
		    bias_vector.push_back(machine->get_bias());
			SG_UNREF(machine);
	    }
	
        return bias_vector;	
    }

    void CMulticlassLogisticRegression::set_w(vector<Eigen::VectorXd>& wnew)
    {
        
        int n_machines = get_num_machines();
        
        for (int32_t i=0; i<n_machines; i++)
	    {
		    CLinearMachine* machine = (CLinearMachine*)m_machines->get_element(i);
		    machine->set_w(SGVector<float64_t>(wnew.at(i)));
			SG_UNREF(machine);
	    }
	
    }

    bool CMulticlassLogisticRegression::train_machine(CFeatures* data)
    {
	    if (data)
		    set_features((CDotFeatures*)data);

		m_machines->reset_array();

	    REQUIRE(m_features, "%s::train_machine(): No features attached!\n");
	    REQUIRE(m_labels, "%s::train_machine(): No labels attached!\n");
	    REQUIRE(m_labels->get_label_type()==LT_MULTICLASS, "%s::train_machine(): "
			    "Attached labels are no multiclass labels\n");
	    REQUIRE(m_multiclass_strategy, "%s::train_machine(): No multiclass strategy"
			    " attached!\n");

	    int32_t n_classes = ((CMulticlassLabels*)m_labels)->get_num_classes();
	    int32_t n_feats = m_features->get_dim_feature_space();

	    slep_options options = slep_options::default_options();
	    options.tolerance = m_epsilon;
	    options.max_iter = m_max_iter;
	    slep_result_t result = slep_mc_plain_lr(m_features,
												(CMulticlassLabels*)m_labels,
												m_z, 
												options);

	    SGMatrix<float64_t> all_w = result.w;
	    SGVector<float64_t> all_c = result.c;

        for (int32_t i=0; i<n_classes; i++)
	    {
		    SGVector<float64_t> w(n_feats);
		    for (int32_t j=0; j<n_feats; j++)
			    w[j] = all_w(j,i);
            
		    float64_t c = all_c[i];
		    CLinearMachine* machine = new CLinearMachine();
		    machine->set_w(w);
		    machine->set_bias(c);
		    m_machines->push_back(machine);
	    }
	    return true;
    }
}
