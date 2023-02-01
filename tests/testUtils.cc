#include "testsHeader.h"

Feat make_estimator(
    int pop_size,
    int gens,
    string ml,
    bool classification,
    int verbosity,
    int random_state
)
{

	Feat feat;
    feat.set_pop_size(pop_size);
    feat.set_gens(gens);
    feat.set_ml(ml);
    feat.set_classification(classification);
    feat.set_verbosity(verbosity);
	feat.set_random_state(random_state);
    feat.set_selection("lexicase");
    feat.set_survival("offspring");
    return feat;
};