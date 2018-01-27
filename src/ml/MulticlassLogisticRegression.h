/* This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef MULTICLASSLOGISTICREGRESSION_H_
#define MULTICLASSLOGISTICREGRESSION_H_
#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMulticlassMachine.h>

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/slep/slep_mc_plain_lr.h>

namespace shogun
{

/** @brief multiclass logistic regression
 *
 */
class CMulticlassLogisticRegression : public CLinearMulticlassMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** default constructor  */
		CMulticlassLogisticRegression();

		/** standard constructor
		 * @param z z regularization constant value
		 * @param feats features
		 * @param labs labels
		 */
		CMulticlassLogisticRegression(float64_t z, CDotFeatures* feats, CLabels* labs);

		/** destructor */
		virtual ~CMulticlassLogisticRegression();

		/** get name */
		virtual const char* get_name() const
		{
			return "MulticlassLogisticRegression";
		}

		/** set z
		 * @param z z value
		 */
		inline void set_z(float64_t z)
		{
			ASSERT(z>0)
			m_z = z;
		}
		/** get C
		 * @return C value
		 */
		inline float64_t get_z() const { return m_z; }

		/** set epsilon
		 * @param epsilon epsilon value
		 */
		inline void set_epsilon(float64_t epsilon)
		{
			ASSERT(epsilon>0)
			m_epsilon = epsilon;
		}
		/** get epsilon
		 * @return epsilon value
		 */
		inline float64_t get_epsilon() const { return m_epsilon; }

		/** set max iter
		 * @param max_iter max iter value
		 */
		inline void set_max_iter(int32_t max_iter)
		{
			ASSERT(max_iter>0)
			m_max_iter = max_iter;
		}
		/** get max iter
		 * @return max iter value
		 */
		inline int32_t get_max_iter() const { return m_max_iter; }

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data = NULL);

private:

		/** init defaults */
		void init_defaults();

		/** register parameters */
		void register_parameters();

protected:

		/** regularization constant for each machine */
		float64_t m_z;

		/** tolerance */
		float64_t m_epsilon;

		/** max number of iterations */
		int32_t m_max_iter;

};
}
/////////////////////////////////////////////////////////////////////////////////////// Definitions


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
}

bool CMulticlassLogisticRegression::train_machine(CFeatures* data)
{
	if (data)
		set_features((CDotFeatures*)data);

	REQUIRE(m_features, "%s::train_machine(): No features attached!\n");
	REQUIRE(m_labels, "%s::train_machine(): No labels attached!\n");
	REQUIRE(m_labels->get_label_type()==LT_MULTICLASS, "%s::train_machine(): "
			"Attached labels are no multiclass labels\n");
	REQUIRE(m_multiclass_strategy, "%s::train_machine(): No multiclass strategy"
			" attached!\n");

	int32_t n_classes = ((CMulticlassLabels*)m_labels)->get_num_classes();
	int32_t n_feats = m_features->get_dim_feature_space();

	slep_options options = slep_options::default_options();
	if (m_machines->get_num_elements()!=0)
	{
		SGMatrix<float64_t> all_w_old(n_feats, n_classes);
		SGVector<float64_t> all_c_old(n_classes);
		for (int32_t i=0; i<n_classes; i++)
		{
			CLinearMachine* machine = (CLinearMachine*)m_machines->get_element(i);
			SGVector<float64_t> w = machine->get_w();
			for (int32_t j=0; j<n_feats; j++)
				all_w_old(j,i) = w[j];
			all_c_old[i] = machine->get_bias();
			SG_UNREF(machine);
		}
		options.last_result = new slep_result_t(all_w_old,all_c_old);
		m_machines->reset_array();
	}
	options.tolerance = m_epsilon;
	options.max_iter = m_max_iter;
	slep_result_t result = slep_mc_plain_lr(m_features,(CMulticlassLabels*)m_labels,m_z,options);

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
#endif
