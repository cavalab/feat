/*
 * edited by William La Cava (WGL), UPenn, 2018 
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


#include "MyCARTreeNodeData.h"

namespace shogun
{

    MyCARTreeNodeData::MyCARTreeNodeData()
    {
	    attribute_id=-1;
	    transit_into_values=SGVector<float64_t>();
	    node_label=-1.0;
	    total_weight=0.;
	    weight_minus_node=0.;
	    weight_minus_branch=0.;
	    num_leaves=0;
        // WGL
        IG = -1.0;    
    }

    void MyCARTreeNodeData::print_data(const MyCARTreeNodeData &data)
    {
	    SG_SPRINT("classifying feature index=%d\n", data.attribute_id);
	    data.transit_into_values.display_vector(data.transit_into_values.vector,data.transit_into_values.vlen, "transit values");
	    SG_SPRINT("total weight=%f\n", data.total_weight);
	    SG_SPRINT("errored weight of node=%f\n", data.weight_minus_node);
	    SG_SPRINT("errored weight of subtree=%f\n", data.weight_minus_branch);
        //WGL
        SG_SPRINT("IG of node=%f\n",data.IG);
	    SG_SPRINT("number of leaves in subtree=%d\n", data.num_leaves);
    }

} /* shogun */

