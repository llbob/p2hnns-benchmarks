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

		void preprocess(int n, int d, int leaf, float *data)
		{
			bc_tree.reset(new BC_Tree<float>(n, d, leaf, data));
		}

		std::vector<int> search(int top_k, int cand, float c, float *query)
		{
			MinK_List list(top_k);
			bc_tree->nns(top_k, cand, c, query, &list);

			std::vector<int> return_list;
			for (int i = 0; i < list.size(); ++i)
			{
				return_list.push_back(list.ith_id(i));
			}
			return return_list;
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