/* FEAT
copyright 2017 William La Cava
authors: William La Cava
license: GNU/GPL v3
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
// json support
#include "pybind11_json/pybind11_json.hpp"
#include "nlohmann/json.hpp"

// py::dict obj = py::dict("number"_a=1234, "hello"_a="world");
// // Automatic py::dict->nl::json conversion
// nl::json j = obj;

// // Automatic nl::json->py::object conversion
// py::object result1 = j;
// // Automatic nl::json->py::dict conversion
// py::dict result2 = j;

#include "feat.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace FT;
namespace nl = nlohmann;
using namespace pybind11::literals;

PYBIND11_MODULE(_feat, m)
{
    m.doc() = R"pbdoc(
        Python binding for Feat
        --------------------------

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    // #ifdef VERSION_INFO
    //     m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    // #else
    //     m.attr("__version__") = "dev";
    // #endif

    py::class_<Feat>(m, "cppFeat", py::dynamic_attr())
        // .def(py::init<>())
        .def(py::init([]()
                      { Feat est; return est; }))
        .def_property("stats_", &Feat::get_stats, nullptr)
        .def_property("pop_size", &Feat::get_pop_size, &Feat::set_pop_size)
        .def_property("gens", &Feat::get_gens, &Feat::set_gens)
        .def_property("ml", &Feat::get_ml, &Feat::set_ml)
        .def_property("classification", &Feat::get_classification, &Feat::set_classification)
        .def_property("verbosity", &Feat::get_verbosity, &Feat::set_verbosity)
        .def_property("max_stall", &Feat::get_max_stall, &Feat::set_max_stall)
        .def_property("sel", &Feat::get_sel, &Feat::set_sel)
        .def_property("surv", &Feat::get_surv, &Feat::set_surv)
        .def_property("cross_rate", &Feat::get_cross_rate, &Feat::set_cross_rate)
        .def_property("root_xo_rate", &Feat::get_root_xo_rate, &Feat::set_root_xo_rate)
        .def_property("otype", &Feat::get_otype, &Feat::set_otype)
        .def_property("max_depth", &Feat::get_max_depth, &Feat::set_max_depth)
        .def_property("max_dim", &Feat::get_max_dim, &Feat::set_max_dim)
        .def_property("random_state", &Feat::get_random_state, &Feat::set_random_state)
        .def_property("erc", &Feat::get_erc, &Feat::set_erc)
        .def_property("objectives", &Feat::get_objectives, &Feat::set_objectives)
        .def_property("functions", &Feat::get_functions, &Feat::set_functions)
        .def_property("shuffle", &Feat::get_shuffle, &Feat::set_shuffle)
        .def_property("split", &Feat::get_split, &Feat::set_split)
        .def_property("fb", &Feat::get_fb, &Feat::set_fb)
        .def_property("scorer", &Feat::get_scorer, &Feat::set_scorer)
        .def_property("feature_names", &Feat::get_feature_names, &Feat::set_feature_names)
        .def_property("backprop", &Feat::get_backprop, &Feat::set_backprop)
        .def_property("iters", &Feat::get_iters, &Feat::set_iters)
        .def_property("lr", &Feat::get_lr, &Feat::set_lr)
        .def_property("batch_size", &Feat::get_batch_size, &Feat::set_batch_size)
        .def_property("n_jobs", &Feat::get_n_jobs, &Feat::set_n_jobs)
        .def_property("hillclimb", &Feat::get_hillclimb, &Feat::set_hillclimb)
        .def_property("logfile", &Feat::get_logfile, &Feat::set_logfile)
        .def_property("max_time", &Feat::get_max_time, &Feat::set_max_time)
        .def_property("residual_xo", &Feat::get_residual_xo, &Feat::set_residual_xo)
        .def_property("stagewise_xo", &Feat::get_stagewise_xo, &Feat::set_stagewise_xo)
        .def_property("stagewise_xo_tol", &Feat::get_stagewise_xo_tol,
                      &Feat::set_stagewise_xo_tol)
        .def_property("softmax_norm", &Feat::get_softmax_norm, &Feat::set_softmax_norm)
        .def_property("save_pop", &Feat::get_save_pop, &Feat::set_save_pop)
        .def_property("normalize", &Feat::get_normalize, &Feat::set_normalize)
        .def_property("val_from_arch", &Feat::get_val_from_arch, &Feat::set_val_from_arch)
        .def_property("corr_delete_mutate", &Feat::get_corr_delete_mutate,
                      &Feat::set_corr_delete_mutate)
        .def_property("simplify", &Feat::get_simplify, &Feat::set_simplify)
        .def_property("protected_groups",
                      &Feat::get_protected_groups, &Feat::set_protected_groups)
        .def_property("tune_initial", &Feat::get_tune_initial, &Feat::set_tune_initial)
        .def_property("tune_final", &Feat::get_tune_final, &Feat::set_tune_final)
        .def_property("starting_pop", &Feat::get_starting_pop, &Feat::set_starting_pop)
        // .def_property("fitted_", &Feat::get_is_fitted, &Feat::set_is_fitted)
        .def("fit",
             py::overload_cast<MatrixXf &, VectorXf &>(&Feat::fit),
             py::call_guard<
                 py::scoped_ostream_redirect,
                 py::scoped_estream_redirect,
                 py::gil_scoped_release>(),
             "fit from X,y data")
        .def("fit",
             py::overload_cast<MatrixXf &, VectorXf &, LongData &>(&Feat::fit),
             py::call_guard<
                 py::scoped_ostream_redirect,
                 py::scoped_estream_redirect,
                 py::gil_scoped_release>(),
             "fit from X,y,Z data")
        .def("transform",
             py::overload_cast<MatrixXf &>(&Feat::transform),
             "transform from X data")
        .def("transform",
             py::overload_cast<MatrixXf &, LongData &>(&Feat::transform),
             "transform from X,Z data")
        .def("predict",
             py::overload_cast<MatrixXf &>(&Feat::predict),
             "predict from X data")
        .def("predict",
             py::overload_cast<MatrixXf &, LongData &>(&Feat::predict),
             "predict from X,Z data")
        .def("predict_proba",
             py::overload_cast<MatrixXf &>(&Feat::predict_proba),
             "predict probabilities from X data")
        .def("predict_proba",
             py::overload_cast<MatrixXf &, LongData &>(&Feat::predict_proba),
             "predict probabilities from X data")
        .def("predict_archive",
             py::overload_cast<int, MatrixXf &>(&Feat::predict_archive),
             "predict from individual in archive")
        .def("predict_archive",
             py::overload_cast<int, MatrixXf &, LongData &>(&Feat::predict_archive),
             "predict from individual in archive")
        .def("predict_proba_archive",
             py::overload_cast<int, MatrixXf &>(&Feat::predict_proba_archive),
             "predict from individual in archive")
        .def("predict_proba_archive",
             py::overload_cast<int, MatrixXf &, LongData &>(&Feat::predict_proba_archive),
             "predict from individual in archive")
        .def("get_archive", &Feat::get_archive, py::arg("front") = false)
        .def("get_coefs", &Feat::get_coefs)
        .def("save", &Feat::save)
        .def("load", &Feat::load)
        .def("get_representation", &Feat::get_representation)
        .def("get_n_params", &Feat::get_n_params)
        .def("get_dim", &Feat::get_dim)
        .def("get_n_nodes", &Feat::get_n_nodes)
        .def("get_model", &Feat::get_model, py::arg("sort") = true)
        .def("get_eqn", &Feat::get_eqn, py::arg("sort") = true)
        ;
    // py::add_ostream_redirect(m, "ostream_redirect");
}
