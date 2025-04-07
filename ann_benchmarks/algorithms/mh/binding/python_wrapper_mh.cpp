#include <baseline.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "util.h"

namespace py = pybind11;

namespace p2h
{
    // int   n,                            // number of data  objects
    // int   d,                            // dimension of space
	// int   M,                            // #proj vecotr used for a single hasher
    // int   m,                            // #single hasher of the compond hasher
    // int   l,                            // #hash tables
    // float b,                            // interval ratio
    // const float *data)                  // input data
	class WrapperMH
	{
	public:
		std::unique_ptr<Angular_Hash<float>> mh;

		void preprocess(int n, int d, int M, int m, int l, float b, py::array_t<float> data)
		{
			py::buffer_info buf = data.request();
			float* ptr = static_cast<float*>(buf.ptr);
			mh.reset(new Angular_Hash<float>(n, d, M, m, l, b, ptr));
		}

		std::tuple<std::vector<int>, int> search(int top_k, int cand, py::array_t<float> query)
		{
			py::buffer_info buf = query.request();
			float* ptr = static_cast<float*>(buf.ptr);
			
			MinK_List list(top_k);
			int num_lin_scans = mh->nns(top_k, cand, ptr, &list);

			std::vector<int> return_list;
			for (int i = 0; i < list.size(); ++i)
			{
				// we need to subtract 1 because the ids are 1-indexed in the C++ code
				return_list.push_back(list.ith_id(i)-1);
			}
			return std::make_tuple(return_list, num_lin_scans);
		}
	};

	PYBIND11_MODULE(mh, m)
	{
		py::class_<WrapperMH>(m, "MH")
			.def(py::init<>())
			.def("preprocess", &WrapperMH::preprocess)
			.def("search", &WrapperMH::search);
	}
}