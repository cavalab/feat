/* FEAT
copyright 2017 William La Cava
authors: William La Cava
license: GNU/GPL v3
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "feat.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace FT;

PYBIND11_MODULE(_feat, m) {
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

    py::class_<Feat>(m, "Feat")
        // .def("fit", &FT::Feat::fit)
        // .def("fit_with_z", &FT::Feat::fit_with_z)
        // .def("predict", &FT::Feat::predict)
        // ;
        // .def("fit",
        //      &Feat::fit,
        //      py::arg("Z") = LongData()
        //      );
        // void fit(MatrixXf& X, VectorXf& y);
        // void fit(MatrixXf& X, VectorXf& y, LongData& Z);
        .def("fit",
             static_cast<void (Feat::*)(MatrixXf& X, VectorXf& y)>(&Feat::fit),
             "fit from X,y data")
        .def("fit",
             static_cast<void (Feat::*)(MatrixXf& X, VectorXf& y, LongData& Z)>(&Feat::fit),
             "fit from X,y,Z data")
        .def("predict",
             static_cast<VectorXf (Feat::*)(MatrixXf& X)>(&Feat::predict),
             "predict from X data")
        ;

}