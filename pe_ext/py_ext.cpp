#include <pybind11/pybind11.h>
namespace py = pybind11;


#include <cmath>

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
	return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
	return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double tanh_impl(double x) {
	return sinh_impl(x) / cosh_impl(x);
}

PYBIND11_MODULE(py_ext, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("tanh_impl", &tanh_impl, "A function which adds two numbers");
}