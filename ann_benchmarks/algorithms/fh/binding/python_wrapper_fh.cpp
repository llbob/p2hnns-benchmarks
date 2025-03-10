#include <fh.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "util.h"
#include "rqalsh.h"

namespace py = pybind11;

namespace p2h
{
    // int   n,                            // number of input data
    // int   d,                            // dimension of input data
    // int   m,                            // #hashers
    // int   s,                            // scale factor of dimension
	// float b,                            // interval ratio
    // const DType *data)                  // input data
	class WrapperFH
	{
	public:
		std::unique_ptr<FH<float>> fh;

		void preprocess(int n, int d, int m, int s, float b, py::array_t<float> data)
		{
			py::buffer_info buf = data.request();
			float* ptr = static_cast<float*>(buf.ptr);
			fh.reset(new FH<float>(n, d, m, s, b, ptr));
		}


		// int   top_k,                    // top-k value
        // int   l,                        // separation threshold (how many buckets it should appear in before becoming a candidate)
        // int   cand,                     // #candidates
        // const float *query,             // input query
        // MinK_List *list);               // top-k results (return)
		std::vector<int> search(int top_k, int l, int cand, py::array_t<float> query)
		{
			py::buffer_info buf = query.request();
			float* ptr = static_cast<float*>(buf.ptr);
			
			MinK_List list(top_k);
			fh->nns(top_k, l, cand, ptr, &list);

			std::vector<int> return_list;
			for (int i = 0; i < list.size(); ++i)
			{
				// we need to subtract 1 because the ids are 1-indexed in the C++ code
				return_list.push_back(list.ith_id(i));
			}
			return return_list;
		}
	};

	PYBIND11_MODULE(fh, m)
	{
		py::class_<WrapperFH>(m, "FH")
			.def(py::init<>())
			.def("preprocess", &WrapperFH::preprocess)
			.def("search", &WrapperFH::search);
	}
}