#include "ann_benchmarks/algorithms/balltree/external_repo/methods/bc_tree.h"
#include <memory>
#include <vector>

namespace p2h {  // Match the namespace used in bc_tree.h

class WrapperBCTree {
public:
    std::unique_ptr<BC_Tree<float>> bc_tree;  // Specify the template parameter
    int n;      // number of data points
    int d;      // dimension
    int leaf;   // leaf size
    float data;    //dataset
    
    WrapperBCTree(int n_, int d_, int leaf_, float data) 
        : n(n_), d(d_), leaf(leaf_), data(data) {}
    
    void preprocess(float* data, int param)
    {
        // Reset the unique_ptr with the new object
        bc_tree.reset(new BC_Tree<float>(n, d, leaf, data));
    }
    
    std::vector<int> search(float* query, int k)
    {
        std::vector<int> return_list;
        MinK_List list(k);  // Create a MinK_List object
        bc_tree->nns(k, query, &list);
        
        // Convert MinK_List to vector<int>
        for (int i = 0; i < list.size(); i++) {
            return_list.push_back(list[i].id_);
        }
        return return_list;
    }
};

}  // namespace p2h