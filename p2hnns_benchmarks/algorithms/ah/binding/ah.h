#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  AH_Hash: Multilinear Hyperplane Hash
// -----------------------------------------------------------------------------
template<class DType>
class AH_Hash : public Basic_Hash<DType> {
public:
    AH_Hash(                        // constructor
        int   n,                        // number of input data 
        int   d,                        // dimension of input data 
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const int   *index,             // index of input data 
        const float *data);             // input data

    // -------------------------------------------------------------------------
    virtual ~AH_Hash();             // destructor

    // -------------------------------------------------------------------------
    virtual int nns(                // point-to-hyperplane NNS
        int   cand,                     // #candidates
        const DType *data,              // input data
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    using SigType = int32_t;

    virtual void get_sig_data(      // get the signature of data
        const float *data,              // input data
        std::vector<SigType> &sig) const; // signature (return)

    virtual void get_sig_query(     // get the signature of query
        const float *query,             // input query 
        std::vector<SigType> &sig) const; // signature (return)

    // -------------------------------------------------------------------------
    virtual uint64_t get_memory_usage() { // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += buckets_.get_memory_usage();
        ret += sizeof(float)*(uint64_t)m_*l_*dim_; // projv
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   m_;                       // #single hasher of the compond hasher
    int   l_;                       // #hash tables
    const int *index_;              // index of input data

    float *projv_;                  // random projection vectors
    KLBucketingFlat<SigType> buckets_; // hash tables

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        const float *data,              // input data
        const float *proj) const;       // random projection vector
};

// -----------------------------------------------------------------------------
template<class DType>
AH_Hash<DType>::AH_Hash(            // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const int   *index,                 // index of input data
    const float *data)                  // input data
    : n_(n), dim_(d), m_(m), l_(l), index_(index), buckets_(n, l)
{
    // sample random projection variables
    uint64_t size = (uint64_t) 2 * m * l * d;
    projv_ = new float[size];
    for (size_t i = 0; i < size; ++i) {
        projv_[i] = gaussian(0.0f, 1.0f);
    }
    // build hash table for the hash values of data objects
    std::vector<SigType> sigs(l);
    for (int i = 0; i < n; ++i) {
        get_sig_data(&data[(uint64_t) i*d], sigs);
        buckets_.insert(i, sigs);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void AH_Hash<DType>::get_sig_data(  // get signature of data
    const float *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < 2 * m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*2*m_+j) * dim_];
            float val = calc_ip<float>(dim_, data, proj);

            SigType new_sig = (val > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
// template<class DType>
// float AH_Hash<DType>::calc_hash_value(// calc hash value
//     const float *data,                  // input data
//     const float *proj) const            // random projection vector
// {
//     float product = 1.0f;
//     for (int i = 0; i < 2; ++i) {
//         float val = calc_ip<float>(dim_, data, &proj[i*dim_]);
//     }
//     return product;
// }

// -----------------------------------------------------------------------------
template<class DType>
AH_Hash<DType>::~AH_Hash()          // destructor
{
    delete[] projv_;
}

// -----------------------------------------------------------------------------
template<class DType>
int AH_Hash<DType>::nns(            // point-to-hyperplane NNS
    int   cand,                         // #candidates
    const DType *data,                  // input data
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);

    int cand_cnt = 0;
    buckets_.for_cand(cand, sigs, [&](int idx) {
        // verify the true distance of idx
        const DType *point = &data[(uint64_t) index_[idx]*dim_];
        float dist = fabs(calc_ip2<DType>(dim_, point, query));
        
        list->insert(dist, index_[idx] + 1);
        ++cand_cnt;
    });
    return cand_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void AH_Hash<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < 2 * m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*2*m_+j) * dim_];
            float val = calc_ip<float>(dim_, query, proj);
            if (j % 2 == 1) {
                val = - val;
            }

            SigType new_sig = (val > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
} // end namespace p2h
