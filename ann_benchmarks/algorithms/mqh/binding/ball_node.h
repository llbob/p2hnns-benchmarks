#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <numeric>
#include <cstddef>

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Ball_Node: leaf node and internal nodea data structure of Ball_Tree
// -----------------------------------------------------------------------------
template<class DType>
class Ball_Node {
public:
    int   n_;                       // number of data points
    int   d_;                       // dimension of data points
    Ball_Node<DType> *lc_;          // left  child
    Ball_Node<DType> *rc_;          // right child
    int   *index_;                  // data index
    DType *data_;                   // data points
    
    float radius_;                  // radius of the ball node
    float *center_;                 // center (the centroid of local data)

    // -------------------------------------------------------------------------
    Ball_Node(                      // constructor
        int   n,                        // number of data points
        int   d,                        // dimension of data points
        bool  is_leaf,                  // is leaf node
        Ball_Node<DType> *lc,           // left  child
        Ball_Node<DType> *rc,           // right child
        int   *index,                   // data index
        const DType *data);             // data points

    // -------------------------------------------------------------------------
    ~Ball_Node();                   // desctructor

    // -------------------------------------------------------------------------
    int nns(                       // point-to-hyperplane nns on ball node
        float c,                        // approximate ratio
        float abso_ip,                  // absolute ip of query & centroid
        float norm_q,                   // the norm of query for d dim
        const float *query,             // input query
        int   &cand,                    // candidate counter (return)
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    float est_lower_bound(          // estimate lower bound
        float c,                        // approximation ratio
        float abso_ip,                  // absolute ip of query & centroid
        float norm_q,                   // the norm of query for d dim
        float radius);                  // radius

    // -------------------------------------------------------------------------
    void add_leaf_points(               // linear scan
        const float *query,             // input query
        float abso_ip,                    // storing distance from center of leaf node to query
        int  &cand,                    // candidate counter (return)
        MinK_List *list);               // top-k PNN results (return)

    // -------------------------------------------------------------------------
    void traversal(                 // traversal ball-tree
        std::vector<int> &leaf_size);   // leaf size (return)
    
    // -------------------------------------------------------------------------
    u64 get_memory_usage() {
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += sizeof(float)*d_; // center_
        if (data_ == nullptr) {  // internal node
            return ret + lc_->get_memory_usage() + rc_->get_memory_usage();
        } else { // leaf node
            return ret;
        }
    }
};

// -----------------------------------------------------------------------------
template<class DType>
Ball_Node<DType>::Ball_Node(        // constructor
    int   n,                            // number of data points
    int   d,                            // data dimension
    bool  is_leaf,                      // is leaf node
    Ball_Node *lc,                      // left  child
    Ball_Node *rc,                      // right child
    int   *index,                       // data index
    const DType *data)                  // data points
    : n_(n), d_(d), lc_(lc), rc_(rc), index_(index), data_(nullptr)
{
    // calc the centroid
    center_ = new float[d];
    calc_centroid<DType>(n, d, index, data, center_);
    
    // calc the radius
    radius_ = -1.0f;
    for (int i = 0; i < n; ++i) {
        const DType *point = data + (u64) index[i]*d;
        float dist = calc_l2_sqr2<DType>(d, point, center_);
        if (dist > radius_) radius_ = dist;
    }
    radius_ = sqrt(radius_);
    
    if (is_leaf) {
        // init local data (only for leaf node)
        data_ = new DType[(u64) n*d];
        for (int i = 0; i < n; ++i) {
            const DType *point = data + (u64) index[i]*d;
            std::copy(point, point+d, data_+(u64)i*d);
        }
    }
}

// -----------------------------------------------------------------------------
template<class DType>
Ball_Node<DType>::~Ball_Node()      // desctructor
{
    if (lc_ != nullptr) { delete lc_; lc_ = nullptr; }
    if (rc_ != nullptr) { delete rc_; rc_ = nullptr; }
    
    if (center_ != nullptr) { delete[] center_; center_ = nullptr; }
    if (data_   != nullptr) { delete[] data_;   data_   = nullptr; }
}

// -----------------------------------------------------------------------------
template<class DType>
int Ball_Node<DType>::nns(         // point-to-hyperplane nns on ball node
    float c,                            // approximate ratio
    float abso_ip,                      // absolute ip of query & centroid
    float norm_q,                       // the norm of query for d dim
    const float *query,                 // input query
    int   &cand,                        // candidate counter (return)
    MinK_List *list)                    // top-k results (return)
{
    // early stop 1
    if (cand <= 0) return 0;
    
    // early stop 2 <- this stops when the current node can't possibly contain points better than what we already have based on a lower bound
    float lower_bound = est_lower_bound(c, abso_ip, norm_q, radius_);
    if (lower_bound >= list->max_key()) return 0;
    
    int num_lin_scans = 0;

    // traversal the tree
    if (data_ != nullptr) { // leaf node
        num_lin_scans++;
        add_leaf_points(query, abso_ip, cand, list);
    } 
    else { // internal node
        // center preference
        // <-- tghis rders the traversal based on the absolute inner product (abso_ip) between the query and node centers
        float lc_abso_ip = fabs(calc_inner_product<float>(d_, lc_->center_, query));
        float rc_abso_ip = fabs(calc_inner_product<float>(d_, rc_->center_, query));
        if (lc_abso_ip < rc_abso_ip) {
            num_lin_scans += lc_->nns(c, lc_abso_ip, norm_q, query, cand, list);
            num_lin_scans += rc_->nns(c, rc_abso_ip, norm_q, query, cand, list);
        }
        else {
            num_lin_scans += rc_->nns(c, rc_abso_ip, norm_q, query, cand, list);
            num_lin_scans += lc_->nns(c, lc_abso_ip, norm_q, query, cand, list);
        }
        
        // // lower bound preference
        // <- would order the traversal based on the estimated lower bounds and visit nodes with smaller lower bound values first
        // float lc_abso_ip = fabs(calc_inner_product<float>(d_, lc_->center_, query));
        // float rc_abso_ip = fabs(calc_inner_product<float>(d_, rc_->center_, query));
        
        // float lc_lb = est_lower_bound(c, lc_abso_ip, norm_q, lc_->radius_);
        // float rc_lb = est_lower_bound(c, rc_abso_ip, norm_q, rc_->radius_);
        // if (lc_lb < rc_lb) {
        //     lc_->nns(c, lc_abso_ip, norm_q, query, cand, list);
        //     rc_->nns(c, rc_abso_ip, norm_q, query, cand, list);
        // }
        // else {
        //     rc_->nns(c, rc_abso_ip, norm_q, query, cand, list);
        //     lc_->nns(c, lc_abso_ip, norm_q, query, cand, list);
        // }
    }
    return num_lin_scans;
}

// -----------------------------------------------------------------------------
template<class DType>
float Ball_Node<DType>::est_lower_bound(// estimate lower bound
    float c,                            // approximation ratio
    float abso_ip,                      // absolute ip of query & centroid
    float norm_q,                       // the norm of query for d dim
    float radius)                       // radius
{
    float threshold = radius * norm_q;
    
    if (abso_ip > threshold) return c*(abso_ip-threshold);
    else return 0.0f;
}

// -----------------------------------------------------------------------------
template<class DType>
void Ball_Node<DType>::add_leaf_points( // linear scan
    const float *query,                 // input query
    float abso_ip,                        // abso_ip which is the abs distance from node center to query
    int  &cand,                    // candidate counter (return)
    MinK_List *list)                    // top-k results (return)
{
    for (int i = 0; i < n_; ++i) {
        // Inser abso_ip calculated from leaf node center to query
        list->insert(abso_ip, index_[i]+1); 
        
        // update candidate counter
        --cand; if (cand <= 0) return;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void Ball_Node<DType>::traversal(   // traversal ball-tree
    std::vector<int> &leaf_size)        // leaf size (return)
{
    if (data_ != nullptr) { // leaf node
        leaf_size.push_back(n_);
    }
    else { // internal node
        lc_->traversal(leaf_size);
        rc_->traversal(leaf_size);
    }
}

} // end namespace p2h
