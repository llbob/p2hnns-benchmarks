#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mqh.h"

namespace py = pybind11;

class MQHWrapper {
private:
    std::unique_ptr<MQH> mqh;
        
public:
    MQHWrapper(int dim, int M2 = 16, int level = 4, int m_level = 1, int m_num = 64) {
        mqh = std::make_unique<MQH>(dim, M2, level, m_level, m_num);
    }
        
    void build_index(py::array_t<float> dataset) {
        // Get input dimensions
        py::buffer_info buf = dataset.request();
            
        if (buf.ndim != 2) {
            throw std::runtime_error("Input dataset must be a 2D array");
        }
            
        int n_pts = buf.shape[0];
        int dim = buf.shape[1];
        float* data_ptr = static_cast<float*>(buf.ptr);
            
        // Convert to 2D vector
        std::vector<std::vector<float>> data_vec(n_pts, std::vector<float>(dim));
        for (int i = 0; i < n_pts; i++) {
            for (int j = 0; j < dim; j++) {
                data_vec[i][j] = data_ptr[i * dim + j];
            }
        }
            
        // Build the index
        mqh->build_index(data_vec);
    }
        
    py::tuple search(py::array_t<float> query, int k, float u, int l0, float delta, int flag) {
        // Get query vector
        py::buffer_info buf = query.request();
            
        if (buf.ndim != 1) {
            throw std::runtime_error("Query must be a 1D array");
        }
            
        float* query_ptr = static_cast<float*>(buf.ptr);
        std::vector<float> query_vec(query_ptr, query_ptr + buf.shape[0]);
            
        // Perform the search (match parameter order with MQH::query)
        std::vector<int> external_candidates;
        auto [results, lin_scans] = mqh->query_with_candidates(query_vec, k, u, l0, delta, flag, external_candidates);
            
        // Convert results to numpy arrays
        std::vector<int> indices;
        std::vector<float> distances;
            
        indices.reserve(results.size());
        distances.reserve(results.size());
            
        for (const auto& res : results) {
            indices.push_back(res.id);
            distances.push_back(res.distance);
        }
            
        return py::make_tuple(py::cast(indices), py::cast(distances));
    }
};

PYBIND11_MODULE(pymqh, m) {
    py::class_<MQHWrapper>(m, "MQH")
        .def(py::init<int, int, int, int, int>(),
            py::arg("dim"), 
            py::arg("M2") = 16, 
            py::arg("level") = 4, 
            py::arg("m_level") = 1, 
            py::arg("m_num") = 64)
        .def("build_index", &MQHWrapper::build_index)
        .def("search", &MQHWrapper::search, 
            py::arg("query"), 
            py::arg("k"), 
            py::arg("b") = 0.0,  // renamed from u to b for clarity
            py::arg("l0") = 3,
            py::arg("delta") = 0.5,
            py::arg("flag") = 0);
}