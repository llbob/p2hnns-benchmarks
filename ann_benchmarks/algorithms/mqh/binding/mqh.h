#ifndef MQH_H
#define MQH_H

#include <vector>
#include <queue>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include "visited_list_pool.h"
#include <iostream>

using namespace std;

// Platform-specific intrinsics setup
#ifdef _MSC_VER
    // Microsoft compiler
    #include <intrin.h>
    #define __builtin_popcount(t) __popcnt(t)
#else
    #if defined(MQH_ARM)
        // ARM-based systems
        #define NO_AVX
        #define NO_SSE
    #else
        // Check for x86 intrinsics
        #if defined(HAVE_X86INTRIN)
            #include <x86intrin.h>
        #elif defined(__AVX__)
            // AVX is supported by compiler
            #include <immintrin.h>
        #elif defined(__SSE2__)
            // SSE2 is supported but not AVX
            #include <emmintrin.h>
            #define NO_AVX
        #else
            // No SIMD extensions detected
            #define NO_AVX
            #define NO_SSE
        #endif
    #endif
#endif

using namespace hnswlib;

// This fast_count is used to count the number of bits that are different between two integers
static inline int fast_count(unsigned long a, unsigned long b) {
    unsigned long u = a ^ b;
#ifdef _MSC_VER
    int count = __popcnt64(u);
#else
    int count = __builtin_popcountll(u);
#endif
    return count;
}

// This function is used to compare float vectors
static float compare_short(const float *a, const float *b, unsigned size) {
    float dot0, dot1, dot2, dot3;
    const float *last = a + size;
    const float *unroll_group = last - 3;
    float result = 0;
    
    while (a < unroll_group) {
        dot0 = a[0] * b[0];
        dot1 = a[1] * b[1];
        dot2 = a[2] * b[2];
        dot3 = a[3] * b[3];
        result += dot0 + dot1 + dot2 + dot3;
        a += 4;
        b += 4;
    }
    
    while (a < last) {
        result += *a++ * *b++;
    }
    
    return result;
}

// The compare_ip function calculates the inner product (dot product) between two float vectors
static float compare_ip(const float *a, const float *b, unsigned size) {
    float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);              \
    tmp2 = _mm256_loadu_ps(addr2);              \
    tmp1 = _mm256_mul_ps(tmp1, tmp2);           \
    dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (size + 7) & ~7U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR) {
        AVX_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        AVX_DOT(l, r, sum, l0, r0);
        AVX_DOT(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_loadu_ps(addr1);              \
    tmp2 = _mm_loadu_ps(addr2);              \
    tmp1 = _mm_mul_ps(tmp1, tmp2);           \
    dest = _mm_add_ps(dest, tmp1);
    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (size + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR) {
    case 12:
        SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
    case 8:
        SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
    case 4:
        SSE_DOT(e_l, e_r, sum, l0, r0);
    default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        SSE_DOT(l, r, sum, l0, r0);
        SSE_DOT(l + 4, r + 4, sum, l1, r1);
        SSE_DOT(l + 8, r + 8, sum, l2, r2);
        SSE_DOT(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
    float dot0, dot1, dot2, dot3;
    const float *last = a + size;
    const float *unroll_group = last - 3;

    while (a < unroll_group) {
        dot0 = a[0] * b[0];
        dot1 = a[1] * b[1];
        dot2 = a[2] * b[2];
        dot3 = a[3] * b[3];
        result += dot0 + dot1 + dot2 + dot3;
        a += 4;
        b += 4;
    }
    while (a < last) {
        result += *a++ * *b++;
    }
#endif
#endif
#else
    // Fallback implementation for non-GCC compilers
    float dot0, dot1, dot2, dot3;
    const float *last = a + size;
    const float *unroll_group = last - 3;

    while (a < unroll_group) {
        dot0 = a[0] * b[0];
        dot1 = a[1] * b[1];
        dot2 = a[2] * b[2];
        dot3 = a[3] * b[3];
        result += dot0 + dot1 + dot2 + dot3;
        a += 4;
        b += 4;
    }
    while (a < last) {
        result += *a++ * *b++;
    }
#endif
    return result;
}

// Basic data structures
struct elem {
    int id;
    float val;
};

struct Q_elem {
    unsigned char id1;
    unsigned char id2;
    int num;
};

struct Neighbor {
    int id;
    float distance;
    
    //This is an overloading of the less than operator to compare two Neighbor objects
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

// Calculate the norm of a vector
static float calc_norm(const float *array, int d) {
    float sum = 0;
    for (int i = 0; i < d; i++) {
        sum += array[i] * array[i];
    }
    return sqrt(sum);
}

// Random number generation
float uniform(float min,float max)  {
	int num = rand();
	float base = (float)RAND_MAX - 1.0F;
	float frac = ((float)num) / base;

	return (max - min) * frac + min;
}

// This function generates a Gaussian random number with a given mean and standard deviation
float gaussian(float mean, float sigma){
	float v1 = -1.0f;
	float v2 = -1.0f;
	float s = -1.0f;
	float x = -1.0f;

	do
	{
		v1 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		v2 = 2.0F * uniform(0.0F, 1.0F) - 1.0F;
		s = v1 * v1 + v2 * v2;
	} while (s >= 1.0F);
	x = v1 * sqrt(-2.0F * log(s) / s);

	x = x * sigma + mean;
	return x;
}

float Quantile(float *table, float a, int size) {
	int i = 0;
	for (i = 0; i < size; i++)
	{
		if (a < table[i])
			break;
	}
	return (1.0f * i / 100);
}


static inline float pq_dist(const unsigned char *a, float **b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x = a[i];
        sum += b[i][x];
    }
    return sum;
}


// inserts a new neighbor into a sorted priority queue while maintaining order.
static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }

    while (left > 0) {
        if (addr[left].distance < nn.distance)
            break;
        if (addr[left].id == nn.id)
            return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
        return K + 1;
    memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

class MQH {
    private:
        const int L = 256; // Number of centroids for quantization
        
        int n_pts;          // Number of data points
        int dim;            // Dimension of data points
        int d_org;          // Original dimension (before padding)
        int d_supp;         // Padding dimension
        int M2;             // Number of subcodebooks
        int level;          // Number of levels
        int m_level;        // Number of hash tables per level
        int m_num;          // Number of bits per hash table
        int flag;           // Flag for precise (1) or approximate (0) search
        
        // Coarse quantization centroids
        std::vector<std::vector<float>> coarse_centroids_first_half;  // First half
        std::vector<std::vector<float>> coarse_centroids_second_half;  // Second half
        
        // PQ centroids for each subspace and level
        std::vector<std::vector<std::vector<float>>> pq_codebooks;
        
        // Random projection vectors for LSH
        std::vector<std::vector<float>> proj_array;
        
        // Quantization cell information
        std::vector<Q_elem> coarse_cell_mapping;
        std::vector<int> count;                     // Points per cell
        std::vector<std::vector<int>> coarse_index; // Point IDs in each cell
        
        // Index data structure
        std::vector<char> index_;
        int size_per_element_;
        
        // Original data
        std::vector<std::vector<float>> data;

        // Constants for probabilistic search guarantees
        const float epsilon = 0.99999; // Desired success probability (very close to 1)
        const float alpha = 0.673; // LSH parameter for controlling collision probability
        
        const float PI = 3.1415926535;

        
        
        void K_means(const std::vector<std::vector<float>>& train, 
                    std::vector<std::vector<float>>& centroids, 
                    int n_sample, int d);
        
        void select_sample(const std::vector<std::vector<float>>& data, 
                          std::vector<std::vector<float>>& train, 
                          int n_pts, int size, int dim);


    
    public:
        MQH(int dim, int M2_ = 16, int level_ = 4, int m_level_ = 1, int m_num_ = 64);
        ~MQH();
        
        void build_index(const std::vector<std::vector<float>>& dataset);
        std::pair<std::vector<Neighbor>, std::vector<int>> query_with_candidates(const std::vector<float>& query_pt, int k, float u, int l0, float delta, int query_flag, std::vector<int>& external_candidates);

        // Getters and setters
        int get_dim() const { return dim; }
        int get_size() const { return n_pts; }
        int get_flag() const { return flag; }
    };

MQH::MQH(int dim_, int M2_, int level_, int m_level_, int m_num_) : 
    dim(dim_), M2(M2_), level(level_), m_level(m_level_), m_num(m_num_) {
    
    d_org = dim;
    
    // Calculate padding if necessary
    if (dim % M2 == 0) {
        d_supp = 0;
    } else {
        d_supp = M2 - dim % M2;
    }
    
    dim = dim + d_supp;  // Padded dimension
    
}

MQH::~MQH() {

}

void MQH::K_means(const std::vector<std::vector<float>>& train, 
                 std::vector<std::vector<float>>& centroids, 
                 int n_sample, int d) {
    
    // Initialize centroids randomly
    int seed_ = 1;
    int cur_obj = 0;
    std::vector<int> array_(L);
    bool flag_ = false; // Flag to check for duplicate samples

    for (int i = 0; i < L; i++) {
        std::srand(seed_);
        seed_++;
        int l = std::rand() % n_sample;
        
        for (int j = 0; j < d; j++) {
            centroids[i][j] = train[l][j];
        }
        
        flag_ = false;
        for (int j = 0; j < cur_obj; j++) {
            if (l == array_[j]) {
                i--;
                flag_ = true;
                break;
            }
        }
        
        if (flag_ == false) {
            array_[cur_obj] = l;
            cur_obj++;
        }
    }

    float sum, min_sum;
    int vec_id;
    std::vector<int> pvec(n_sample);
    std::vector<int> cluster_sizes(L, 0); // Number of points in each cluster
    int ROUND = 20;

    // K-means iterations
    for (int k = 0; k < ROUND; k++) {
        // Fill cluster_sizes with zeros
        std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);

        // Assign each point to nearest centroid
        for (int j = 0; j < n_sample; j++) {
            for (int l = 0; l < L; l++) {
                sum = 0;
                for (int i = 0; i < d; i++) {
                    sum += (train[j][i] - centroids[l][i]) * (train[j][i] - centroids[l][i]);
                }
                
                if (l == 0) {
                    min_sum = sum;
                    vec_id = 0;
                } else if (sum < min_sum) {
                    min_sum = sum;
                    vec_id = l;
                }
            }
            
            pvec[j] = vec_id;
            cluster_sizes[pvec[j]]++;
        }

        // Reset centroids
        for (int j = 0; j < L; j++) {
            for (int i = 0; i < d; i++) {
                centroids[j][i] = 0;
            }
        }

        // Update centroids with mean of assigned points
        for (int j = 0; j < n_sample; j++) {
            for (int i = 0; i < d; i++) {
                centroids[pvec[j]][i] += train[j][i];
            }
        }

        // Compute final centroids
        for (int j = 0; j < L; j++) {
            if (cluster_sizes[j] == 0)
                continue;
                
            for (int i = 0; i < d; i++) {
                centroids[j][i] = centroids[j][i] / cluster_sizes[j];
            }
        }
    }
}

// Use an interval based on a sample size and a dataset size in order to pick sample points for a training set of points
void MQH::select_sample(const std::vector<std::vector<float>>& data, 
                      std::vector<std::vector<float>>& train, 
                      int n_pts, int size, int dim) {
    
    int interval = n_pts / size;
    int cur_obj = 0;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < dim; j++) {
            train[i][j] = data[cur_obj][j];
        }
        cur_obj += interval;
    }
}


void MQH::build_index(const std::vector<std::vector<float>>& dataset) {
    n_pts = dataset.size();
    
    // Allocate and initialize data with padding
    data.resize(n_pts, std::vector<float>(dim, 0));
    for (int i = 0; i < n_pts; i++) {
        for (int j = 0; j < d_org; j++) {
            data[i][j] = dataset[i][j];
        }
    }
    
    int size = M2 * level;  // Total number of subcodebooks
    int num_hash_functions = m_level * m_num;  // Total number of hash functions
    
    int sample_size = 100000; // Samples used for training k-means

    // Calculate norms of data points
    std::vector<float> norm(n_pts);
    for (int i = 0; i < n_pts; i++) {
        norm[i] = calc_norm(data[i].data(), dim);
    }
    
    // Prepare for quantization
    std::vector<std::vector<float>> residual_vec(n_pts, std::vector<float>(dim));
    
    // Initialize centroids for coarse quantization
    coarse_centroids_first_half.resize(L, std::vector<float>(dim / 2));
    coarse_centroids_second_half.resize(L, std::vector<float>(dim / 2));
    
    // Sample training data
    int n_sample = std::min(sample_size, n_pts);
    std::vector<std::vector<float>> train(n_sample, std::vector<float>(dim));
    select_sample(data, train, n_pts, n_sample, dim);
    
    // Split training data for the two coarse quantizers
    std::vector<std::vector<float>> train1(n_sample, std::vector<float>(dim / 2));
    std::vector<std::vector<float>> train2(n_sample, std::vector<float>(dim / 2));
    
    for (int i = 0; i < n_sample; i++) {
        // First half of dimensions
        for (int j = 0; j < dim / 2; j++) {
            train1[i][j] = train[i][j];
        }
        // Second half of dimensions
        for (int j = 0; j < dim / 2; j++) {
            train2[i][j] = train[i][j + dim / 2];
        }
    }
    
    // Run K-means for coarse quantization
    K_means(train1, coarse_centroids_first_half, n_sample, dim / 2);
    K_means(train2, coarse_centroids_second_half, n_sample, dim / 2);
    
    // Arrays for coarse quantization assignments
    std::vector<unsigned char> centroid_ids_first_half(n_pts);
    std::vector<unsigned char> centroid_ids_second_half(n_pts);
    std::vector<int> count_all(L * L, 0);
    
    // Initialize data structures for assigning points to cells
	float min_sum;
	int min_id;
    // Assign data points to coarse quantization cells
    for (int i = 0; i < n_pts; i++) {
        // Find closest centroid for first half
        
        for (int j = 0; j < L; j++) {
            float sum = 0;
            for (int l = 0; l < dim / 2; l++) {
                sum += (data[i][l] - coarse_centroids_first_half[j][l]) * (data[i][l] - coarse_centroids_first_half[j][l]);
            }
            
            if (j == 0) {
                min_sum = sum;
                min_id = 0;
            } else if (sum < min_sum) {
                min_sum = sum;
                min_id = j;
            }
        }
        centroid_ids_first_half[i] = min_id;
        
        // Find closest centroid for second half
        for (int j = 0; j < L; j++) {
            float sum = 0;
            for (int l = 0; l < dim / 2; l++) {
                sum += (data[i][l + dim / 2] - coarse_centroids_second_half[j][l]) * (data[i][l + dim / 2] - coarse_centroids_second_half[j][l]);
            }
            
            if (j == 0) {
                min_sum = sum;
                min_id = 0;
            } else if (sum < min_sum) {
                min_sum = sum;
                min_id = j;
            }
        }
        centroid_ids_second_half[i] = min_id;
        
        // Increment count for this combination
        count_all[centroid_ids_first_half[i] * L + centroid_ids_second_half[i]]++;
    }
    
    // Count non-empty cells and create mapping
    int num_nonempty_cells = 0;
    std::vector<int> map_table(L * L, -1);
    
    for (int i = 0; i < L * L; i++) {
        if (count_all[i] > 0) {
            num_nonempty_cells++;
        }
    }
    
    // Initialize data structures for non-empty cells
    std::vector<std::vector<elem>> cell_to_point_ids(num_nonempty_cells);
    std::vector<int> n_temp(num_nonempty_cells, 0);
    coarse_cell_mapping.resize(num_nonempty_cells);
    count.resize(num_nonempty_cells);
    
    // Create compact mapping
    num_nonempty_cells = 0;
    for (int i = 0; i < L * L; i++) {
        if (count_all[i] > 0) {
            cell_to_point_ids[num_nonempty_cells].resize(count_all[i]);
            map_table[i] = num_nonempty_cells;
            
            coarse_cell_mapping[num_nonempty_cells].id1 = i / L;
            coarse_cell_mapping[num_nonempty_cells].id2 = i % L;
            coarse_cell_mapping[num_nonempty_cells].num = count_all[i];
            count[num_nonempty_cells] = count_all[i];
            
            num_nonempty_cells++;
        }
    }
    
    // Compute residual vectors by way of reconstructing the vectors from their quantized forms
    std::vector<float> reconstructed_vector(dim);
    
    for (int i = 0; i < n_pts; i++) {
        int temp = centroid_ids_first_half[i] * L + centroid_ids_second_half[i];
        int table_id = map_table[temp];
        
        cell_to_point_ids[table_id][n_temp[table_id]].id = i;
        
        // Reconstruct quantized vector
        for (int j = 0; j < dim / 2; j++) {
            reconstructed_vector[j] = coarse_centroids_first_half[centroid_ids_first_half[i]][j];
            reconstructed_vector[j + dim / 2] = coarse_centroids_second_half[centroid_ids_second_half[i]][j];
        }
        
        // Compute residual (original - quantized)
        for (int j = 0; j < dim; j++) {
            residual_vec[i][j] = data[i][j] - reconstructed_vector[j];
        }
        
        // Calculate residual norm
        float residual_norm = 0;
        for (int j = 0; j < dim; j++) {
            residual_norm += residual_vec[i][j] * residual_vec[i][j];
        }
        
        cell_to_point_ids[table_id][n_temp[table_id]].val = std::sqrt(residual_norm);
        n_temp[table_id]++;
    }
    
    // Sort points in each cell by residual norm
    for (int i = 0; i < num_nonempty_cells; i++) {
        std::sort(cell_to_point_ids[i].begin(), cell_to_point_ids[i].end(), 
                 [](const elem& a, const elem& b) { return a.val < b.val; });
    }
    
    // Calculate residual norms
    std::vector<float> norm2(n_pts);
    for (int i = 0; i < n_pts; i++) {
        norm2[i] = calc_norm(residual_vec[i].data(), dim);
    }
    
    // Normalize residual vectors
    std::vector<bool> zero_flag(n_pts, false);
    float min_float = 0.0000001;
    
    for (int i = 0; i < n_pts; i++) {
        if (norm2[i] < min_float) {
            zero_flag[i] = true;
            residual_vec[i][0] = 1;
            for (int j = 1; j < dim; j++) {
                residual_vec[i][j] = 0;
            }
        } else {
            for (int j = 0; j < dim; j++) {
                residual_vec[i][j] = residual_vec[i][j] / norm2[i];
            }
        }
    }
    
    // Storage for PQ codes
    std::vector<std::vector<unsigned char>> pq_id(n_pts, std::vector<unsigned char>(M2));
    
    // Generate random projection vectors for LSH
    proj_array.resize(num_hash_functions, std::vector<float>(dim));
    for (int i = 0; i < num_hash_functions; i++) {
        for (int j = 0; j < dim; j++) {
            proj_array[i][j] = gaussian(0.0f, 1.0f);
        }
    }
    
    // // Compute binary hash codes for normalized residual vectors
    // std::vector<std::vector<unsigned long>> bin_hash_codes(n_pts, std::vector<unsigned long>(m_level));
    
    // for (int i = 0; i < n_pts; i++) {
    //     for (int j = 0; j < m_level; j++) {
    //         unsigned long code_num = 0;
    //         for (int l = 0; l < m_num; l++) {
    //             float ip_with_proj_vec = 0;
    //             for (int ll = 0; ll < dim; ll++) {
    //                 ip_with_proj_vec += residual_vec[i][ll] * proj_array[j * m_num + l][ll];
    //             }
                
    //             if (ip_with_proj_vec >= 0) {
    //                 code_num += 1;
    //             }
                
    //             if (l < m_num - 1) {
    //                 code_num = code_num << 1;
    //             }
    //         }
    //         bin_hash_codes[i][j] = code_num;
    //     }
    // }
    
// =======================================================================================================================
// Prepare compact index structure by writing coarse level info
    
    size_per_element_ = 2 * sizeof(unsigned char) + 
    level * (2 * sizeof(float) + M2 + sizeof(unsigned long) * m_level);

    // REMEMBER :
    //    size_per_element_ = sizeof(int) +                                 // Point ID
    //    sizeof(float) +                                                   // Coarse level residual norm 
    //    sizeof(float) +                                                   // Additional float value (VAL) to be used later
    //    2 * sizeof(unsigned char) +                                       // Coarse centroid IDs (1 byte each)
    //    level * (M2 + 2 * sizeof(float) + sizeof(unsigned long) * m_level);   // Level data to be filled out later. M2 = PQ_IDs, residual norm = sizeof(float), hashcode = sizeof(unsigned long)

    // Initialize index structure for each cluster
    coarse_index.resize(num_nonempty_cells);
    index_.resize(n_pts * size_per_element_);

    for (int i = 0; i < num_nonempty_cells; i++) {
        coarse_index[i].resize(count[i]);
        
        // Store point IDs and residual norms
        for (int j = 0; j < count[i]; j++) {
            int point_id = cell_to_point_ids[i][j].id;
            float residual_norm = cell_to_point_ids[i][j].val;
            
            //starting point of data for this point
            char* cur_loc = &index_[point_id * size_per_element_];
            
            // //Store point ID
            // memcpy(cur_loc, &point_id, sizeof(int));
            // cur_loc += sizeof(int);
            
            //store residual norm
            // memcpy(cur_loc, &residual_norm, sizeof(float));
            // cur_loc += sizeof(float);
            
            // //space for VAL to be used later
            // float val_placeholder = 0.0f;
            // memcpy(cur_loc, &val_placeholder, sizeof(float));
            // cur_loc += sizeof(float);
            
            
            coarse_index[i][j] = cell_to_point_ids[i][j].id;
            
            // Write coarse centroid IDs
            
            unsigned char centroid_id_first = coarse_cell_mapping[i].id1;  // Get from cell mapping
            unsigned char centroid_id_second = coarse_cell_mapping[i].id2;
            
            // wrtie first coarse centroid ID
            memcpy(cur_loc, &centroid_id_first, sizeof(unsigned char));
            cur_loc += sizeof(unsigned char);
            // write second coarse centroid ID
            memcpy(cur_loc, &centroid_id_second, sizeof(unsigned char));
        }
    }

    // =======================================================================================================================
    // Begin multilevel product quantization while writing level data to index

    // Initialize product quantization centroids
    int M2_dim = dim / M2;
    pq_codebooks.resize(size, std::vector<std::vector<float>>(L, std::vector<float>(M2_dim)));

    // Prepare for multilevel PQ
    std::vector<std::vector<float>> pq_training_samples(n_sample, std::vector<float>(M2_dim));
    std::vector<float> relative_norms(n_pts);

    //outer loop for the PQ and LSH-code calculations at each level
    for (int k = 0; k < level; k++) {
        // For each subspace, train quantizers, on residual subvectors
        for (int i = 0; i < M2; i++) {
            int sample_count = 0;
            // get training samples for this subspace
            for (int j = 0; j < n_pts; j++) {
                if (zero_flag[j] == true) {
                    continue;
                }
                for (int l = 0; l < M2_dim; l++) {
                    pq_training_samples[sample_count][l] = residual_vec[j][i * M2_dim + l];
                }
                sample_count++;
                if (sample_count >= n_sample) {
                    break;
                }
            }
            
            // K-means for this subspace
            K_means(pq_training_samples, pq_codebooks[k * M2 + i], sample_count, M2_dim);
        }

    // Assign each point to closest centroid in each subspace
    for (int n = 0; n < n_pts; n++) {
        for (int i = 0; i < M2; i++) {
            float min_sum;
            unsigned char min_id;
            
            // find closest centroid 
            for (int j = 0; j < L; j++) {
                float sum = 0;
                for (int l = 0; l < M2_dim; l++) {
                    float diff = residual_vec[n][i * M2_dim + l] - pq_codebooks[k * M2 + i][j][l];
                    sum += diff * diff;
                }
                
                if (j == 0) {
                    min_sum = sum;
                    min_id = 0;
                } else if (sum < min_sum) {
                    min_sum = sum;
                    min_id = j;
                }
            }
            pq_id[n][i] = min_id;
        }
    }

    for (int n = 0; n < n_pts; n++) {
        //reconstruct vector from PQ codes
        std::vector<float> reconstructed(dim, 0.0f);
        for (int i = 0; i < M2; i++) {
            for (int l = 0; l < M2_dim; l++) {
                reconstructed[i * M2_dim + l] = pq_codebooks[k * M2 + i][pq_id[n][i]][l];
            }
        }
        
        // norm of reconstructed vector
        float centroid_norm = calc_norm(reconstructed.data(), dim);
        
        // relative norm 
        relative_norms[n] = norm2[n] / centroid_norm;
        
        
        // reconstructed vector is scaled by original residual norm
        for (int j = 0; j < dim; j++) {
            reconstructed[j] *= relative_norms[n];
        }
        
        // new residual
        for (int j = 0; j < dim; j++) {
            residual_vec[n][j] -= reconstructed[j];
        }
        
        // norm of new residual
        float sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += residual_vec[n][j] * residual_vec[n][j];
        }
        norm2[n] = std::sqrt(sum);
        
        // activate zero flag if below threshold
        if(norm2[n] < min_float && zero_flag[n] == false) {
            zero_flag[n] = true;
            residual_vec[n][0] = 1;
            for (int j = 1; j < dim; j++) {
                residual_vec[n][j] = 0;
            }
        }
        else {
            // normalize new residual for next level's NERQ quantization
            for(int j = 0; j < dim; j++) {
                residual_vec[n][j] /= norm2[n];
            }
        }
    }

    // generate hash codes for this level
    std::vector<std::vector<unsigned long>> level_hash_codes(n_pts, std::vector<unsigned long>(m_level));

    for (int i = 0; i < n_pts; i++) {
        for (int j = 0; j < m_level; j++) {
            unsigned long code_num = 0;
            for (int l = 0; l < m_num; l++) {
                float ip_with_proj_vec = 0;
                for (int ll = 0; ll < dim; ll++) {
                    ip_with_proj_vec += residual_vec[i][ll] * proj_array[j * m_num + l][ll];
                }
                
                if (ip_with_proj_vec >= 0) {
                    code_num += 1;
                }
                
                if (l < m_num - 1) {
                    code_num = code_num << 1;
                }
            }
            level_hash_codes[i][j] = code_num;
        }
    }

    // store this leveløs hash codes and codewords in index
    for (int n = 0; n < n_pts; n++) {
        // position pointer for this level's data
        char* cur_loc = &index_[n * size_per_element_ + 2 * sizeof(unsigned char) + 
                    k * (2 * sizeof(float) + M2 + sizeof(unsigned long) * m_level)];
        
                    
        // write relative residual norm 
        memcpy(cur_loc, &relative_norms[n], sizeof(float));
        cur_loc += sizeof(float);
        
        // write actual residual norm at this level
        memcpy(cur_loc, &norm2[n], sizeof(float));
        cur_loc += sizeof(float);

        // write PQ codes 
        for (int l = 0; l < M2; l++) {
            memcpy(cur_loc, &pq_id[n][l], 1);
            cur_loc += 1;
        }

        // write hash codes 
        for (int l = 0; l < m_level; l++) {
            memcpy(cur_loc, &level_hash_codes[n][l], sizeof(unsigned long));
            cur_loc += sizeof(unsigned long);
        }
        }
    }
}


std::pair<std::vector<Neighbor>, std::vector<int>> MQH::query_with_candidates(const std::vector<float>& query_pt, int k, float u, int l0, float delta, int query_flag, std::vector<int>& external_candidates) {
    if (static_cast<int>(query_pt.size()) != d_org) {
        throw std::runtime_error("Query dimension doesn't match index dimension");
    }
    
    //____________________________________________________________________________________________________________________________________
    // Preprocess query arguments and set variables

    // use passed flag, 0 for efficiency, 1 for recall guarantees
    int FLAG = query_flag;
    int num_linear_scans = 0;
    
    // Pad query if necessary
    std::vector<float> query(dim, 0);
    for (int i = 0; i < d_org; i++) {
        query[i] = query_pt[i];
    }
    
    // Normalize query
    float query_norm = calc_norm(query.data(), dim);
    for (int i = 0; i < dim; i++) {
        query[i] /= query_norm;
    }
    
    // U is the bias. Divides u by the query norm to maintain the same geometric relationship andnegates the value for the distance calculation
    float b = -1 * u / query_norm;
    
    //____________________________________________________________________________________________________________________________________
    // Precompute inner products of coarse centroids with q

    std::vector<float> first_half_ips(L);
    std::vector<float> second_half_ips(L);

    //first half
    int half_dim = dim/2;
    for (int j = 0; j < L; j++) {
        first_half_ips[j] = compare_short(query.data(), coarse_centroids_first_half[j].data(), half_dim);
    }

    //second half
    for (int j = 0; j < L; j++) {
        second_half_ips[j] = compare_short(query.data() + dim / 2, coarse_centroids_second_half[j].data(), half_dim);
    }

    // if no external initial candidate selection, the algorithm resorts to bruteforce distance calcs of coarse centroids
    if(external_candidates.size() == 0) {
    //calculate distance from query to each coarse quantization cell
        std::vector<std::pair<int, float>> cell_distances;
        cell_distances.reserve(coarse_cell_mapping.size());

        for (int j = 0; j < static_cast<int>(coarse_cell_mapping.size()); j++) {
            unsigned char a = coarse_cell_mapping[j].id1;
            unsigned char b = coarse_cell_mapping[j].id2;
            
            float dist = first_half_ips[a] + second_half_ips[b] - u;
            cell_distances.push_back(std::make_pair(j, dist));
        }

        //sort cells by distance
        std::sort(cell_distances.begin(), cell_distances.end(),
                [](const std::pair<int,float>& a, const std::pair<int,float>& b) {
                    return std::abs(a.second) < std::abs(b.second);
                });
        
        //we take 1/20 of the total points.
        int cap = n_pts;
        external_candidates.reserve(cap);
        //populate external candidates vector until cap is reached
        for(auto pair : cell_distances) {
            if(external_candidates.size() >= cap) {
                break;
            }
            int centroid_id = pair.first;
            auto points = coarse_index[centroid_id];
            for(int j = 0; j < points.size(); j++){
                external_candidates.push_back(points[j]);
            }


        }
    }
    //____________________________________________________________________________________________________________________________________
    // Precompute inner products of sub space centroids at remaining levels

    std::vector<std::vector<std::vector<float>>> level_ip(
        level, std::vector<std::vector<float>>(M2, std::vector<float>(L)));

    int sub_dim = dim/M2;
    for (int j = 0; j < level; j++) {
        for (int l = 0; l < M2; l++) {
            for (int k = 0; k < L; k++) {
                level_ip[j][l][k] = compare_short(query.data() + l * sub_dim, 
                pq_codebooks[j * M2 + l][k].data(), sub_dim);
            }
        }
    }
    //____________________________________________________________________________________________________________________________________
    //calculate LSH bit string for query both in the positive and negative case

    unsigned long query_bit_string_pos = 0;
    unsigned long query_bit_string_neg = 0;
    for (int l = 0; l < m_num; l++) {
        float positive_q_ip = 0;
        float negative_q_ip = 0;
        for (int ll = 0; ll < dim; ll++) {
            positive_q_ip += query.data()[ll] * proj_array[l][ll];
            negative_q_ip += -1 * query.data()[ll] * proj_array[l][ll];
        }
        
        if (positive_q_ip >= 0) {
            query_bit_string_pos += 1;
        }
        if (negative_q_ip >= 0) {
            query_bit_string_neg += 1;
        }
        
        if (l < m_num - 1) {
            query_bit_string_pos = query_bit_string_pos << 1;
            query_bit_string_neg = query_bit_string_neg << 1;
        }
    }
    //____________________________________________________________________________________________________________________________________
    // Initialize and populate candidate_set

    std::vector<Neighbor> candidate_set(k); // result set containing k elements that are updated throughout the pruning process.
    for (int i = 0; i < k; i++) {
        candidate_set[i].id = -1;  // Invalid ID
        candidate_set[i].distance = std::numeric_limits<float>::max();
    }
    
    float cur_val = 0.0; // current kth nearest neighbour's distance to H

    //populate running candidate set and set initial cur_val.
    for(int i = 0; i < k; i++)
    {
        int point_id = external_candidates.back();
        external_candidates.pop_back();
        float distance = compare_short(data[point_id].data(), query.data(), dim) - b;
        if (distance < 0) {
            distance = -1 * distance;
        }
        num_linear_scans++;

        //Create neighbor instance
        Neighbor nn;
        nn.id = point_id;
        nn.distance = distance;

        //add to candidate set at right location in PQ
        InsertIntoPool(candidate_set.data(), k, nn);

        //update current kth nearest neighbor distance if needed
        if (distance > cur_val)
        {
            cur_val = distance;
        }
    }

    //====================================================================================================================================
    // Init counters for logging pruning
    int break_condition_1 = 0;
    int break_condition_2 = 0;
    int break_condition_3 = 0;
    int collision_runs = 0;
    int collision_passed = 0;

    // Begin MQH pruning process starting by the outer for loop in pseudocode
    int n = 0;
    for(int point_id : external_candidates) {
        n++;
        // skip point id for now
        char *cur_loc = &index_[point_id * size_per_element_];
        // find coarse centroid IDs and initialize IP by looking up the precomputed ip in each sub space.
        unsigned char first_coarse_id = *reinterpret_cast<unsigned char*>(cur_loc);
        cur_loc += sizeof(unsigned char);
        unsigned char second_coarse_id = *reinterpret_cast<unsigned char*>(cur_loc);
        cur_loc += sizeof(unsigned char);
        float ip = first_half_ips[first_coarse_id] + second_half_ips[second_coarse_id];

        
        // gradual refinement of quantization
        for(int l = 0; l < level; l++){
            //find the right memory location for the point at this level by skipping data already read. Therefore, offset = coarse data + previous levels' data
            int offset = 2 * sizeof(unsigned char) + l * (2 * sizeof(float) + M2 + sizeof(unsigned long));
            cur_loc = &index_[point_id * size_per_element_] + offset;
            // initialize current relative/actual residual norms. 
            float relative_residual_norm = *reinterpret_cast<float*>(cur_loc);
            cur_loc += sizeof(float);
            float actual_residual_norm = *reinterpret_cast<float*>(cur_loc);
            cur_loc += sizeof(float);
            // first update the inner product based on centroid at this level
            // if(n % 1000 == 0) {
            //     cout << "relative norm: " << relative_residual_norm << " ";
            //     cout << "actual residual norm: " << actual_residual_norm;
            // }
            for(int i = 0; i < M2; i++)
            {
                // read one codeword at a time and add corresponding precomputed ip to running ip
                unsigned char codeword = *reinterpret_cast<unsigned char*>(cur_loc);
                ip += level_ip[l][i][codeword] * relative_residual_norm;
                cur_loc += sizeof(unsigned char);
            }
            
            if (ip > b - cur_val && ip < b + cur_val) {
                // Centroid lies within boundaries, so x is a promising candidate who's exact distance to H we calculate
                float dist_to_H = compare_short(data[point_id].data(), query.data(), dim) - b;
                if (dist_to_H < 0) {
                    dist_to_H = - dist_to_H;
                }
                num_linear_scans++;
                
                if(dist_to_H < cur_val)
                {
                    Neighbor nn;
                    nn.id = point_id;
                    nn.distance = dist_to_H;
                    InsertIntoPool(candidate_set.data(), k, nn);
                    cur_val = candidate_set[k-1].distance;
                    break;
                }
            }
            
            // Boolean to check which side of the hyperplane the centroid is situated on
            bool positive_side = u > 0 ? ip < b - cur_val : (ip < 0 || ip < b - cur_val);
            
            float centroid_dist_to_boundary = fabs(ip - b) - cur_val;
            // if(n % 1000 == 0) {
            //     cout << "centroid distance to boundary: " << centroid_dist_to_boundary << endl << endl;
            // }

            if (centroid_dist_to_boundary > actual_residual_norm) // LINE 10 in pseudocode
            {
                // distance from centroid to bouondary is greater than residual norm, so residual cannot by any means reach inside the margin.
                break_condition_1++;
                break;
            }

            if (FLAG == 0 && centroid_dist_to_boundary > actual_residual_norm * delta) { // LINE 12 in pseudocode
                // ratio between centroid's distance to boundary and residual_norm is too large, so we prune for efficiency
                break_condition_2++;
                break;
            }

            if ((FLAG == 1 && centroid_dist_to_boundary > actual_residual_norm * delta) || (FLAG == 0 && l==level-1 && centroid_dist_to_boundary <= actual_residual_norm * delta)) // LINE 14 in pseudocode
            {
                collision_runs++;
                // Collision testing : 

                //First establish bucket with t_zero, t_one, P_zero and P_one
                float t_zero = centroid_dist_to_boundary/actual_residual_norm;
                if (t_zero > 1.0) {
                    t_zero = 1.0;
                }
                float t_one = (centroid_dist_to_boundary + 2 * cur_val)/actual_residual_norm;
                if (t_one > 1.0) {
                    t_one = 1.0;
                }

                float P_zero = 1 - (acos(t_zero)/PI);
                float P_one = 1 - (acos(t_one)/PI);

                int l1 = l0/2;
                int lower_collision_boundary = P_zero * m_num - l1;
                int upper_collision_boundary = P_one * m_num + l1;
                
                //Then read stored bit string for given point at given level
                unsigned long point_bit_string = *reinterpret_cast<unsigned long*>(cur_loc);
                
                // get collision number between query and point
                int collision_number;
                if(positive_side) {
                    collision_number = m_num - fast_count(point_bit_string, query_bit_string_pos);
                }
                else {
                    collision_number = m_num - fast_count(point_bit_string, query_bit_string_neg);
                }
                // if (n % 1000 == 0) { 
                //     cout << "lower collision boundary:" << lower_collision_boundary << " ";
                //     cout << "collision number:" << collision_number << endl << endl;
                // }

                if(collision_number >= lower_collision_boundary && collision_number <= upper_collision_boundary) {
                    collision_passed++;
                    float dist_to_H = compare_short(data[point_id].data(), query.data(), dim) - b;
                    if (dist_to_H < 0) {
                        dist_to_H = - dist_to_H;
                    }
                    num_linear_scans++;

                    if(dist_to_H < cur_val)
                    {
                        Neighbor nn;
                        nn.id = point_id;
                        nn.distance = dist_to_H;
                        InsertIntoPool(candidate_set.data(), k, nn);
                        cur_val = candidate_set[k-1].distance;
                        break;
                    }
                }
                else {
                    break_condition_3++;
                    break;
                }
            }
        }
    }
    // create vector<int> counters to return num_linear_scans and break conditions
    std::vector<int> counters = {num_linear_scans, break_condition_2, break_condition_3, collision_runs, collision_passed};
    // std::vector<Neighbor> results(candidate_set.begin(), candidate_set.begin() + k);

    std::vector<Neighbor> results;
    results.reserve(k);
    for (int i = 0; i < k; i++) {
        if (candidate_set[i].id >= 0) {
            results.push_back(candidate_set[i]);
        }
    }
    // return results
    return {results, counters};
}        
             
#endif // MQH_H