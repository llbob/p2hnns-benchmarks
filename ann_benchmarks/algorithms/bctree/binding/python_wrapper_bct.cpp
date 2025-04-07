#include <bc_tree.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace p2h
{

	class WrapperBCTree
	// int   n_;                       // number of data points
	// int   d_;                       // dimension of data points
	// int   leaf_;                    // leaf size of bc-tree
	// const DType *data_;             // data points
	{
	public:
		std::unique_ptr<BC_Tree<float>> bc_tree;

		void preprocess(int n, int d, int leaf, py::array_t<float> data)
		{
			py::buffer_info buf = data.request();
			float* ptr = static_cast<float*>(buf.ptr);
			bc_tree.reset(new BC_Tree<float>(n, d, leaf, ptr));
		}

		std::tuple<std::vector<int>, int> search(int top_k, int cand, float c, py::array_t<float> query)
		{
			py::buffer_info buf = query.request();
			float* ptr = static_cast<float*>(buf.ptr);
			
			MinK_List list(top_k);
			int num_lin_scans = bc_tree->nns(top_k, cand, c, ptr, &list);

			std::vector<int> return_list;
			for (int i = 0; i < list.size(); ++i)
			{
				// we need to subtract 1 because the ids are 1-indexed in the C++ code
				return_list.push_back(list.ith_id(i)-1);
			}
			return std::make_tuple(return_list, num_lin_scans);
		}
	};

	PYBIND11_MODULE(bc_tree, m)
	{
		py::class_<WrapperBCTree>(m, "BCTree")
			.def(py::init<>())
			.def("preprocess", &WrapperBCTree::preprocess)
			.def("search", &WrapperBCTree::search);
	}
}