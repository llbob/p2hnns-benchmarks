#include <nh.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "util.h"

namespace py = pybind11;

namespace p2h
{
    // int   n,                            // number of input data
    // int   d,                            // dimension of input data
    // int   m,                            // #hashers
    // int   s,                            // scale factor of dimension
    // float w,                            // bucket width
    // const DType *data)                  // input data
	class WrapperNH
	{
	public:
		std::unique_ptr<NH<float>> nh;

		void preprocess(int n, int d, int m, int s, float w, py::array_t<float> data)
		{
			py::buffer_info buf = data.request();
			float* ptr = static_cast<float*>(buf.ptr);
			nh.reset(new NH<float>(n, d, m, s, w, ptr));
		}
		// int   top_k,                        // top-k value
		// int   cand,                         // #candidates
		// const float *query,                 // input query
		// MinK_List *list)                    // top-k results (return)

		std::vector<int> search(int top_k, int cand, py::array_t<float> query)
		{
			py::buffer_info buf = query.request();
			float* ptr = static_cast<float*>(buf.ptr);
			
			MinK_List list(top_k);
			nh->nns(top_k, cand, ptr, &list);

			std::vector<int> return_list;
			for (int i = 0; i < list.size(); ++i)
			{
				// we need to subtract 1 because the ids are 1-indexed in the C++ code
				return_list.push_back(list.ith_id(i));
			}
			return return_list;
		}
	};

	PYBIND11_MODULE(nh, m)
	{
		py::class_<WrapperNH>(m, "NH")
			.def(py::init<>())
			.def("preprocess", &WrapperNH::preprocess)
			.def("search", &WrapperNH::search);
	}
}