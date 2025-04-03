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

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount(t) __popcnt(t)
#else
#if defined(MQH_ARM)
#define NO_AVX
#define NO_SSE
#else
#ifdef HAVE_X86INTRIN
#include <x86intrin.h>
#else
#define AVX
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
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

// Comparison functions for sorting
static int Elemcomp_a(const void *a, const void *b) {
    elem x1 = *((elem *)b);
    elem x2 = *((elem *)a);

    if (x1.val > x2.val) return -1;
    else return 1;
}

static int Elemcomp_d(const void *a, const void *b) {
    elem x1 = *((elem *)b);
    elem x2 = *((elem *)a);

    if (x1.val > x2.val) return 1;
    else return -1;
}

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

// Calculate product quantization distance
static inline float pq_dist(const unsigned char *a, float **b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x = a[i];
        sum += b[i][x];
    }
    return sum;
}

// Priority queue operations
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
        std::vector<std::vector<float>> vec_1;  // First half
        std::vector<std::vector<float>> vec_2;  // Second half
        
        // PQ centroids for each subspace and level
        std::vector<std::vector<std::vector<float>>> vec_pq;
        
        // Random projection vectors for LSH
        std::vector<std::vector<float>> proj_array;
        
        // Quantization cell information
        std::vector<Q_elem> pq_M2;
        std::vector<int> count;                     // Points per cell
        std::vector<std::vector<int>> coarse_index; // Point IDs in each cell
        
        // Index data structure
        std::vector<std::vector<char>> index_;
        int size_per_element_;
        
        // Original data
        std::vector<std::vector<float>> data;
        
        // For search
        VisitedListPool* visited_list_pool_;

        // Constants for probabilistic search guarantees
        const float epsilon = 0.99999; // Desired success probability (very close to 1)
        const float alpha = 0.673; // LSH parameter for controlling collision probability
        
        // Lookup table for quantiles
        std::vector<float> quantile_table;
        
        // Initialize the quantile table
        void init_quantile_table() {
            quantile_table.resize(170);
            quantile_table[0] = 0.5;
            quantile_table[1] = 0.504;
            quantile_table[2] = 0.508;
            quantile_table[3] = 0.512;
            quantile_table[4] = 0.516;
            quantile_table[5] = 0.52;
            quantile_table[6] = 0.524;
            quantile_table[7] = 0.528;
            quantile_table[8] = 0.532;
            quantile_table[9] = 0.536;
        
            quantile_table[10] = 0.54;
            quantile_table[11] = 0.544;
            quantile_table[12] = 0.548;
            quantile_table[13] = 0.552;
            quantile_table[14] = 0.556;
            quantile_table[15] = 0.56;
            quantile_table[16] = 0.564;
            quantile_table[17] = 0.568;
            quantile_table[18] = 0.571;
            quantile_table[19] = 0.575;
        
            quantile_table[20] = 0.58;
            quantile_table[21] = 0.583;
            quantile_table[22] = 0.587;
            quantile_table[23] = 0.591;
            quantile_table[24] = 0.595;
            quantile_table[25] = 0.599;
            quantile_table[26] = 0.603;
            quantile_table[27] = 0.606;
            quantile_table[28] = 0.61;
            quantile_table[29] = 0.614;
        
            quantile_table[30] = 0.618;
            quantile_table[31] = 0.622;
            quantile_table[32] = 0.626;
            quantile_table[33] = 0.63;
            quantile_table[34] = 0.633;
            quantile_table[35] = 0.637;
            quantile_table[36] = 0.641;
            quantile_table[37] = 0.644;
            quantile_table[38] = 0.648;
            quantile_table[39] = 0.652;
        
            quantile_table[40] = 0.655;
            quantile_table[41] = 0.659;
            quantile_table[42] = 0.663;
            quantile_table[43] = 0.666;
            quantile_table[44] = 0.67;
            quantile_table[45] = 0.674;
            quantile_table[46] = 0.677;
            quantile_table[47] = 0.681;
            quantile_table[48] = 0.684;
            quantile_table[49] = 0.688;
        
            quantile_table[50] = 0.692;
            quantile_table[51] = 0.695;
            quantile_table[52] = 0.699;
            quantile_table[53] = 0.702;
            quantile_table[54] = 0.705;
            quantile_table[55] = 0.709;
            quantile_table[56] = 0.712;
            quantile_table[57] = 0.716;
            quantile_table[58] = 0.719;
            quantile_table[59] = 0.722;
        
            quantile_table[60] = 0.726;
            quantile_table[61] = 0.729;
            quantile_table[62] = 0.732;
            quantile_table[63] = 0.736;
            quantile_table[64] = 0.74;
            quantile_table[65] = 0.742;
            quantile_table[66] = 0.745;
            quantile_table[67] = 0.749;
            quantile_table[68] = 0.752;
            quantile_table[69] = 0.755;
        
            quantile_table[70] = 0.758;
            quantile_table[71] = 0.761;
            quantile_table[72] = 0.764;
            quantile_table[73] = 0.767;
            quantile_table[74] = 0.77;
            quantile_table[75] = 0.773;
            quantile_table[76] = 0.776;
            quantile_table[77] = 0.779;
            quantile_table[78] = 0.782;
            quantile_table[79] = 0.785;
        
            quantile_table[80] = 0.788;
            quantile_table[81] = 0.791;
            quantile_table[82] = 0.794;
            quantile_table[83] = 0.797;
            quantile_table[84] = 0.8;
            quantile_table[85] = 0.802;
            quantile_table[86] = 0.805;
            quantile_table[87] = 0.808;
            quantile_table[88] = 0.811;
            quantile_table[89] = 0.813;
        
            quantile_table[90] = 0.816;
            quantile_table[91] = 0.819;
            quantile_table[92] = 0.821;
            quantile_table[93] = 0.824;
            quantile_table[94] = 0.826;
            quantile_table[95] = 0.829;
            quantile_table[96] = 0.832;
            quantile_table[97] = 0.834;
            quantile_table[98] = 0.837;
            quantile_table[99] = 0.839;
        
            quantile_table[100] = 0.841;
            quantile_table[101] = 0.844;
            quantile_table[102] = 0.846;
            quantile_table[103] = 0.849;
            quantile_table[104] = 0.851;
            quantile_table[105] = 0.853;
            quantile_table[106] = 0.855;
            quantile_table[107] = 0.858;
            quantile_table[108] = 0.85;
            quantile_table[109] = 0.862;
        
            quantile_table[110] = 0.864;
            quantile_table[111] = 0.867;
            quantile_table[112] = 0.869;
            quantile_table[113] = 0.871;
            quantile_table[114] = 0.873;
            quantile_table[115] = 0.875;
            quantile_table[116] = 0.877;
            quantile_table[117] = 0.879;
            quantile_table[118] = 0.881;
            quantile_table[119] = 0.883;
        
            quantile_table[120] = 0.885;
            quantile_table[121] = 0.887;
            quantile_table[122] = 0.889;
            quantile_table[123] = 0.891;
            quantile_table[124] = 0.893;
            quantile_table[125] = 0.894;
            quantile_table[126] = 0.896;
            quantile_table[127] = 0.898;
            quantile_table[128] = 0.9;
            quantile_table[129] = 0.902;
        
            quantile_table[130] = 0.903;
            quantile_table[131] = 0.905;
            quantile_table[132] = 0.907;
            quantile_table[133] = 0.908;
            quantile_table[134] = 0.910;
            quantile_table[135] = 0.912;
            quantile_table[136] = 0.913;
            quantile_table[137] = 0.915;
            quantile_table[138] = 0.916;
            quantile_table[139] = 0.918;
        
            quantile_table[140] = 0.919;
            quantile_table[141] = 0.921;
            quantile_table[142] = 0.922;
            quantile_table[143] = 0.924;
            quantile_table[144] = 0.925;
            quantile_table[145] = 0.927;
            quantile_table[146] = 0.928;
            quantile_table[147] = 0.929;
            quantile_table[148] = 0.931;
            quantile_table[149] = 0.932;
        
            quantile_table[150] = 0.933;
            quantile_table[151] = 0.935;
            quantile_table[152] = 0.936;
            quantile_table[153] = 0.937;
            quantile_table[154] = 0.938;
            quantile_table[155] = 0.939;
            quantile_table[156] = 0.941;
            quantile_table[157] = 0.942;
            quantile_table[158] = 0.943;
            quantile_table[159] = 0.944;
        
            quantile_table[160] = 0.945;
            quantile_table[161] = 0.946;
            quantile_table[162] = 0.947;
            quantile_table[163] = 0.948;
            quantile_table[164] = 0.95;
            quantile_table[165] = 0.951;
            quantile_table[166] = 0.952;
            quantile_table[167] = 0.953;
            quantile_table[168] = 0.954;
            quantile_table[169] = 0.955;
        }
        // Precomputed tables for query optimization
        int max_inv;
        std::vector<unsigned char> tab_inv;
        float temp;
        float coeff2;
        float max_coeff;
        std::vector<float> ttab;
        const float PI = 3.1415926535;

        
        
        void K_means(const std::vector<std::vector<float>>& train, 
                    std::vector<std::vector<float>>& centroids, 
                    int n_sample, int d);
        
        void select_sample(const std::vector<std::vector<float>>& data, 
                          std::vector<std::vector<float>>& train, 
                          int n_pts, int size, int dim);

        void init_query_tables();

    
    public:
        MQH(int dim, int M2_ = 16, int level_ = 4, int m_level_ = 1, int m_num_ = 64);
        ~MQH();
        
        void build_index(const std::vector<std::vector<float>>& dataset);
        std::vector<Neighbor> query(const std::vector<float>& query_pt, int k, float u, int l0, float delta, int query_flag);

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
    int m = m_level * m_num;  // Total number of hash functions
    
    // Initialize visited list pool
    visited_list_pool_ = new VisitedListPool(1, 1);  // Will be resized in build_index
    
    // Initialize the quantile table
    init_quantile_table();
    
    // Initialize precomputed query tables
    init_query_tables();
}

MQH::~MQH() {
    delete visited_list_pool_;
}

void MQH::K_means(const std::vector<std::vector<float>>& train, 
                 std::vector<std::vector<float>>& centroids, 
                 int n_sample, int d) {
    
    // Initialize centroids randomly
    int seed_ = 1;
    int cur_obj = 0;
    std::vector<int> array_(L);
    bool flag_ = false;

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
    std::vector<int> count(L, 0);
    int ROUND = 20;

    // K-means iterations
    for (int k = 0; k < ROUND; k++) {
        // Reset counts
        std::fill(count.begin(), count.end(), 0);

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
            count[pvec[j]]++;
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
            if (count[j] == 0)
                continue;
                
            for (int i = 0; i < d; i++) {
                centroids[j][i] = centroids[j][i] / count[j];
            }
        }
    }
}

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

void MQH::init_query_tables() {
    // Calculate temperature parameter for bit sampling based on probability theory
    int m = m_level * m_num;
    temp = std::sqrt(std::log(1.0f / epsilon) / 2.0f / m);
    
    // Calculate lower bound coefficient for bit matching threshold
    coeff2 = Quantile(quantile_table.data(), 0.5f * temp + 0.75f, quantile_table.size());
    
    // Calculate upper bound coefficient for maximum allowable bit differences
    max_coeff = Quantile(quantile_table.data(), 0.5f * (1.0f - temp) + 0.5f, quantile_table.size());
    
    // Create a lookup table for fast mapping between distance ratios and bit thresholds
    max_inv = 1000; // Number of discretized intervals
    tab_inv.resize(max_inv);
    
    // Calculate step size for interpolating between coefficients
    float ratio = (max_coeff - coeff2) / max_inv;
    for (int i = 0; i < max_inv; i++) {
        // Interpolate between min and max coefficients
        float temp2 = coeff2 + i * ratio;
        
        // Map to closest entry in quantile table
        int temp3 = static_cast<int>(temp2 * 100);
        if (temp3 >= static_cast<int>(quantile_table.size()))
            temp3 = quantile_table.size() - 1;
        
        // Convert quantile value to Hamming distance threshold
        temp2 = 2.0f * (quantile_table[temp3] - 0.5f) + temp;
        
        // Convert to actual bit count and ensure it doesn't exceed total bits
        tab_inv[i] = static_cast<unsigned char>(temp2 * m + 1);
        if (tab_inv[i] > m)
            tab_inv[i] = m;
    }
    
    // Create another lookup table for direct quantile-to-bits conversion
    ttab.resize(quantile_table.size());
    for (int i = 0; i < static_cast<int>(quantile_table.size()); i++) {
        // Convert normalized quantile values to bit differences
        ttab[i] = (quantile_table[i] - 0.5f) * 2.0f * m;
    }
}

void MQH::build_index(const std::vector<std::vector<float>>& dataset) {
    n_pts = dataset.size();
    
    // Resize the visited list pool
    delete visited_list_pool_;
    visited_list_pool_ = new VisitedListPool(1, n_pts);
    
    // Allocate and initialize data with padding
    data.resize(n_pts, std::vector<float>(dim, 0));
    for (int i = 0; i < n_pts; i++) {
        for (int j = 0; j < d_org; j++) {
            data[i][j] = dataset[i][j];
        }
    }
    
    int size = M2 * level;  // Total number of subcodebooks
    int m = m_level * m_num;  // Total number of hash functions
    
    int sample_size = 100000; // Samples used for training k-means

    // Calculate norms of data points
    std::vector<float> norm(n_pts);
    for (int i = 0; i < n_pts; i++) {
        norm[i] = calc_norm(data[i].data(), dim);
    }
    
    // Prepare for quantization
    std::vector<std::vector<float>> residual_vec(n_pts, std::vector<float>(dim));
    
    // Initialize centroids for coarse quantization
    vec_1.resize(L, std::vector<float>(dim / 2));
    vec_2.resize(L, std::vector<float>(dim / 2));
    
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
    K_means(train1, vec_1, n_sample, dim / 2);
    K_means(train2, vec_2, n_sample, dim / 2);
    
    // Arrays for coarse quantization assignments
    std::vector<unsigned char> vec_id1(n_pts);
    std::vector<unsigned char> vec_id2(n_pts);
    std::vector<int> count_all(L * L, 0);
    
    // Initialize data structures for assigning points to cells
    float sum = 0;
	float min_sum;
	int min_id;
    // Assign data points to coarse quantization cells
    for (int i = 0; i < n_pts; i++) {
        // Find closest centroid for first half
        
        for (int j = 0; j < L; j++) {
            float sum = 0;
            for (int l = 0; l < dim / 2; l++) {
                sum += (data[i][l] - vec_1[j][l]) * (data[i][l] - vec_1[j][l]);
            }
            
            if (j == 0) {
                min_sum = sum;
                min_id = 0;
            } else if (sum < min_sum) {
                min_sum = sum;
                min_id = j;
            }
        }
        vec_id1[i] = min_id;
        
        // Find closest centroid for second half
        for (int j = 0; j < L; j++) {
            float sum = 0;
            for (int l = 0; l < dim / 2; l++) {
                sum += (data[i][l + dim / 2] - vec_2[j][l]) * (data[i][l + dim / 2] - vec_2[j][l]);
            }
            
            if (j == 0) {
                min_sum = sum;
                min_id = 0;
            } else if (sum < min_sum) {
                min_sum = sum;
                min_id = j;
            }
        }
        vec_id2[i] = min_id;
        
        // Increment count for this combination
        count_all[vec_id1[i] * L + vec_id2[i]]++;
    }
    
    // Count non-empty cells and create mapping
    int n_cand1 = 0;
    std::vector<int> map_table(L * L, -1);
    
    for (int i = 0; i < L * L; i++) {
        if (count_all[i] > 0) {
            n_cand1++;
        }
    }
    
    // Initialize data structures for non-empty cells
    std::vector<std::vector<elem>> array1(n_cand1);
    std::vector<int> n_temp(n_cand1, 0);
    pq_M2.resize(n_cand1);
    count.resize(n_cand1);
    
    // Create compact mapping
    n_cand1 = 0;
    for (int i = 0; i < L * L; i++) {
        if (count_all[i] > 0) {
            array1[n_cand1].resize(count_all[i]);
            map_table[i] = n_cand1;
            
            pq_M2[n_cand1].id1 = i / L;
            pq_M2[n_cand1].id2 = i % L;
            pq_M2[n_cand1].num = count_all[i];
            count[n_cand1] = count_all[i];
            
            n_cand1++;
        }
    }
    
    // Compute residual vectors
    std::vector<float> vec(dim);
    
    for (int i = 0; i < n_pts; i++) {
        int temp = vec_id1[i] * L + vec_id2[i];
        int table_id = map_table[temp];
        
        array1[table_id][n_temp[table_id]].id = i;
        
        // Reconstruct quantized vector
        for (int j = 0; j < dim / 2; j++) {
            vec[j] = vec_1[vec_id1[i]][j];
            vec[j + dim / 2] = vec_2[vec_id2[i]][j];
        }
        
        // Compute residual (original - quantized)
        for (int j = 0; j < dim; j++) {
            residual_vec[i][j] = data[i][j] - vec[j];
        }
        
        // Calculate residual norm
        float residual_norm = 0;
        for (int j = 0; j < dim; j++) {
            residual_norm += residual_vec[i][j] * residual_vec[i][j];
        }
        
        array1[table_id][n_temp[table_id]].val = std::sqrt(residual_norm);
        n_temp[table_id]++;
    }
    
    // Sort points in each cell by residual norm
    for (int i = 0; i < n_cand1; i++) {
        std::sort(array1[i].begin(), array1[i].end(), 
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
    
    // Initialize product quantization centroids
    int d2 = dim / M2;
    vec_pq.resize(size, std::vector<std::vector<float>>(L, std::vector<float>(d2)));
    
    // Storage for PQ codes
    std::vector<std::vector<unsigned char>> pq_id(n_pts, std::vector<unsigned char>(M2));
    
    // Generate random projection vectors for LSH
    proj_array.resize(m, std::vector<float>(dim));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dim; j++) {
            proj_array[i][j] = gaussian(0.0f, 1.0f);
        }
    }
    
    // Compute binary hash codes for normalized residual vectors
    std::vector<std::vector<unsigned long>> rough_code(n_pts, std::vector<unsigned long>(m_level));
    
    for (int i = 0; i < n_pts; i++) {
        for (int j = 0; j < m_level; j++) {
            unsigned long code_num = 0;
            for (int l = 0; l < m_num; l++) {
                float ssum = 0;
                for (int ll = 0; ll < dim; ll++) {
                    ssum += residual_vec[i][ll] * proj_array[j * m_num + l][ll];
                }
                
                if (ssum >= 0) {
                    code_num += 1;
                }
                
                if (l < m_num - 1) {
                    code_num = code_num << 1;
                }
            }
            rough_code[i][j] = code_num;
        }
    }
    
    // Prepare for multilevel PQ
    std::vector<std::vector<float>> residual_pq(n_sample, std::vector<float>(d2));
    
    // Begin multilevel product quantization
    for (int k = 0; k < level; k++) {
        // For each subspace, train quantizers on residual subvectors
        for (int i = 0; i < M2; i++) {
            int ccount = 0;
            // Collect training samples for this subspace
            for (int j = 0; j < n_pts; j++) {
                if (zero_flag[j] == true) {
                    continue;
                }
                for (int l = 0; l < d2; l++) {
                    residual_pq[ccount][l] = residual_vec[j][i * d2 + l];
                }
                ccount++;
                if (ccount >= n_sample) {
                    break;
                }
            }
            
            // Run K-means for this subspace
            K_means(residual_pq, vec_pq[k * M2 + i], ccount, d2);
        }
        
        // Assign each point to closest centroid in each subspace
        for (int n = 0; n < n_pts; n++) {
            for (int i = 0; i < M2; i++) {
                float min_sum;
                int min_id;
                
                // Find closest centroid from the L options
                for (int j = 0; j < L; j++) {
                    float sum = 0;
                    for (int l = 0; l < d2; l++) {
                        float diff = residual_vec[n][i * d2 + l] - vec_pq[k * M2 + i][j][l];
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
        
        // Compute new residuals by subtracting quantized vectors
        for (int n = 0; n < n_pts; n++) {
            for (int j = 0; j < M2; j++) {
                int temp_M = k * M2 + j;
                for (int l = 0; l < d2; l++) {
                    residual_vec[n][j * d2 + l] -= vec_pq[temp_M][pq_id[n][j]][l];
                }
            }
            
            // Calculate norm of new residual vector
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += residual_vec[n][j] * residual_vec[n][j];
            }
            norm2[n] = std::sqrt(sum);
            
            // Flag vectors with negligible residuals
            if (norm2[n] < min_float) {
                zero_flag[n] = true;
            }
        }
    }
    
    // Prepare compact index structure
    size_per_element_ = sizeof(int) + 2 * sizeof(float) + sizeof(unsigned long) * m_level + 
                       level * (M2 + sizeof(float) + sizeof(unsigned long) * m_level);
    
    // Initialize index structure for each cluster
    coarse_index.resize(n_cand1);
    index_.resize(n_cand1);
    
    for (int i = 0; i < n_cand1; i++) {
        coarse_index[i].resize(count[i]);
        index_[i].resize(count[i] * size_per_element_);
        
        // Store point IDs and residual norms
        for (int j = 0; j < count[i]; j++) {
            coarse_index[i][j] = array1[i][j].id;
            
            // Write point ID
            memcpy(&index_[i][j * size_per_element_], &array1[i][j].id, sizeof(int));
            
            // Write residual norm
            memcpy(&index_[i][j * size_per_element_ + sizeof(int)], &array1[i][j].val, sizeof(float));
        }
    }
    
    // Copy hash codes to index
    for (int i = 0; i < n_cand1; i++) {
        for (int j = 0; j < count[i]; j++) {
            int point_id = coarse_index[i][j];
            char* cur_loc = &index_[i][j * size_per_element_ + sizeof(int) + 2 * sizeof(float)];
            
            // Copy hash codes
            for (int jj = 0; jj < m_level; jj++) {
                memcpy(cur_loc, &rough_code[point_id][jj], sizeof(unsigned long));
                cur_loc += sizeof(unsigned long);
            }
        }
    }
    
    // Store PQ codes and norms for each level
    for (int level_idx = 0; level_idx < level; level_idx++) {
        for (int i = 0; i < n_cand1; i++) {
            for (int j = 0; j < count[i]; j++) {
                int point_id = coarse_index[i][j];
                
                // Position pointer for this level's data
                char* cur_loc = &index_[i][j * size_per_element_ + sizeof(int) + 2 * sizeof(float) + 
                               sizeof(unsigned long) * m_level + 
                               level_idx * (M2 + sizeof(float) + sizeof(unsigned long) * m_level)];
                
                // Write residual norm for this level
                memcpy(cur_loc, &norm2[point_id], sizeof(float));
                cur_loc += sizeof(float);
                
                // Write PQ codes for this level
                for (int l = 0; l < M2; l++) {
                    memcpy(cur_loc, &pq_id[point_id][l], 1);
                    cur_loc += 1;
                }
            }
        }
    }
}

std::vector<Neighbor> MQH::query(const std::vector<float>& query_pt, int k, float u, int l0, float delta, int query_flag) {
    if (static_cast<int>(query_pt.size()) != d_org) {
        throw std::runtime_error("Query dimension doesn't match index dimension");
    }
    
    // Constants for search
    int topk = k;
    int n_exact = 2000;  // First-pass candidates
    int thres_pq = n_pts / 10;  // Early termination threshold

    // Use passed flag if valid, otherwise use the instance flag
    int current_flag = (query_flag != -1) ? query_flag : flag;
    
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
    
    // Adjust threshold
    u = -1 * u / query_norm;
    
    // Adjust threshold
    float delta_flag = (current_flag == 1) ? 1.0f : delta;  // For guarantees vs. approximation
    int offset0 = l0;
    
    // Prepare for hash bit comparison
    int m = m_level * m_num;
    
    // Get a visited list for tracking processed points
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    
    // Prepare result sets
    std::vector<Neighbor> retset(topk + 1);
    std::vector<Neighbor> retset2(n_exact + 1);
    
    // Cosine table is computed at query time, the rest of the tables are precomputed at index time
    int cosine_inv = 100;
    std::vector<int> cosine_table(cosine_inv);

    for (int i = 0; i < cosine_inv; i++) {
        cosine_table[i] = m * acos(1.0f * i / cosine_inv) / PI + offset0;
        if (cosine_table[i] > m) {
            cosine_table[i] = m;
        }
    }

    // Compute LSH hash codes for the query
    std::vector<unsigned long> query_proj(m_level);
    
    for (int j = 0; j < m_level; j++) {
        query_proj[j] = 0;
        for (int jj = 0; jj < m_num; jj++) {
            float ttmp0 = compare_ip(query.data(), proj_array[j * m_num + jj].data(), dim);
            
            if (ttmp0 >= 0) {
                query_proj[j] += 1;
            }
            
            if (jj < m_num - 1) {
                query_proj[j] = query_proj[j] << 1;
            }
        }
    }
    
    // Precompute distances to coarse quantizer centroids
    std::vector<float> table_1(L);
    std::vector<float> table_2(L);
    
    // First half
    for (int j = 0; j < L; j++) {
        table_1[j] = compare_short(query.data(), vec_1[j].data(), dim / 2);
    }
    
    // Second half
    for (int j = 0; j < L; j++) {
        table_2[j] = compare_short(query.data() + dim / 2, vec_2[j].data(), dim / 2);
    }
    
    // Precompute distances to PQ centroids
    std::vector<std::vector<std::vector<float>>> table2(
        level, std::vector<std::vector<float>>(M2, std::vector<float>(L)));
    
    for (int j = 0; j < level; j++) {
        for (int l = 0; l < M2; l++) {
            for (int k = 0; k < L; k++) {
                table2[j][l][k] = compare_short(query.data() + l * (dim / M2), 
                                               vec_pq[j * M2 + l][k].data(), dim / M2);
            }
        }
    }
    
    // Calculate distance from query to each coarse quantization cell
    std::vector<std::pair<int, float>> cell_distances;
    cell_distances.reserve(pq_M2.size());
    
    for (int j = 0; j < static_cast<int>(pq_M2.size()); j++) {
        unsigned char a = pq_M2[j].id1;
        unsigned char b = pq_M2[j].id2;
        
        float dist = table_1[a] + table_2[b] - u;
        cell_distances.push_back(std::make_pair(j, dist));
    }
    
    // Sort cells by distance
    std::sort(cell_distances.begin(), cell_distances.end(),
             [](const std::pair<int,float>& a, const std::pair<int,float>& b) {
                 return std::abs(a.second) < std::abs(b.second);
             });
    
    // First-pass: collect candidates
    int num = 0;   // Counter for final results
    int num1 = 0;  // Counter for first-pass candidates
    int points_examined = 0;
    
    // Calculate offset for accessing PQ codes in memory layout
    int offset00 = sizeof(float) + m_level * sizeof(unsigned long);
    int offset1 = m_level * sizeof(unsigned long);
    int offset2 = M2 + m_level * sizeof(unsigned long);
    int offset3 = sizeof(float) + M2 + m_level * sizeof(unsigned long);
    int round1_offset = sizeof(float) + sizeof(unsigned long) * m_level + sizeof(float);
    int remain_size = size_per_element_ - sizeof(int) - (3 * sizeof(float)) - 
                     (m_level * sizeof(unsigned long));
    
    // Process cells in order of distance
    for (const auto& cell_dist : cell_distances) {
        int cell_idx = cell_dist.first;
        float cur_dist = cell_dist.second;
        int cell_size = count[cell_idx];
        
        // Process each point in this cell
        for (int l = 0; l < cell_size; l++) {
            points_examined++;
            
            // Early termination if examined too many points
            if (points_examined > thres_pq) {
                break;
            }
            
            // Get point data
            char* cur_obj = &index_[cell_idx][l * size_per_element_];
            int point_id = *reinterpret_cast<int*>(cur_obj);
            cur_obj += sizeof(int);
            
            float residual_norm = *reinterpret_cast<float*>(cur_obj);
            char* cur_obj_1 = cur_obj + sizeof(float);
            cur_obj = cur_obj_1 + round1_offset;
            
            // Calculate PQ distance
            // Create an array of pointers to table rows
            std::vector<float*> table_ptrs(M2);
            for (int i = 0; i < M2; i++) {
                table_ptrs[i] = table2[0][i].data();
            }

            // Call pq_dist with the array of pointers
            float pq_distance = pq_dist(reinterpret_cast<unsigned char*>(cur_obj), 
                                        table_ptrs.data(), M2);
            cur_obj += remain_size;
            
            // Combine coarse and fine distances -> The fact that we multiply by residual norm here is important, this would be related to NERQ
            float distance = cur_dist + pq_distance * residual_norm;
            
            // Store for later use
            *reinterpret_cast<float*>(cur_obj_1) = distance;
            
            // Use absolute distance
            if (distance < 0) {
                distance = -distance;
            }
            
            // Add to candidate set
            Neighbor nn;
            nn.id = point_id;
            nn.distance = distance;
            
            if (num1 == 0) {
                retset2[0] = nn;
            } else {
                if (num1 >= n_exact) {
                    if (distance >= retset2[n_exact - 1].distance) {
                        continue;
                    }
                    InsertIntoPool(retset2.data(), n_exact, nn);
                } else {
                    InsertIntoPool(retset2.data(), num1, nn);
                }
            }
            num1++;
        }
        
        // Break if we've examined enough points
        if (points_examined > thres_pq) {
            break;
        }
    }
    
    // Second-pass: Refine top candidates with exact distance calculation
    for (int j = 0; j < std::min(n_exact, num1); j++) {
        int point_id = retset2[j].id;
        
        // Skip if already processed
        if (visited_array[point_id] == visited_array_tag) {
            continue;
        }
        
        visited_array[point_id] = visited_array_tag;
        
        // Calculate exact distance
        float distance = compare_ip(data[point_id].data(), query.data(), dim) - u;
        if (distance < 0) {
            distance = -distance;
        }
        
        // Add to final result set
        Neighbor nn;
        nn.id = point_id;
        nn.distance = distance;
        
        if (num == 0) {
            retset[0] = nn;
        } else {
            if (num >= topk) {
                if (distance >= retset[topk - 1].distance) {
                    continue;
                }
                InsertIntoPool(retset.data(), topk, nn);
            } else {
                InsertIntoPool(retset.data(), num, nn);
            }
        }
        num++;
    }
    
    // Third-pass: Process remaining clusters with adaptive filtering
    points_examined = 0;
    float cur_val = retset[num < topk ? num - 1 : topk - 1].distance;  // Current distance threshold
    bool thres_flag = false;
    
    // Process remaining cells
    for (const auto& cell_dist : cell_distances) {
        int cell_idx = cell_dist.first;
        float cell_dist_val = cell_dist.second;
        float v = cell_dist_val;  // Cell distance
        int cell_size = count[cell_idx];
        
        // Track processing progress for early termination
        if (!thres_flag) {
            if (points_examined + cell_size >= thres_pq) {
                thres_flag = true;
            } else {
                points_examined += cell_size;
            }
        }
        
        // Skip entire cell if distance exceeds threshold significantly
        if (std::abs(cell_dist_val) > cur_val + delta_flag * cell_dist_val) {
            continue;
        }
        
        // Examine each point in this cell
        for (int l = 0; l < cell_size; l++) {
            if (thres_flag) {
                points_examined++;
            }
            
            char* cur_obj = &index_[cell_idx][l * size_per_element_];
            int point_id = *reinterpret_cast<int*>(cur_obj);
            
            // Skip if already processed
            if (visited_array[point_id] == visited_array_tag) {
                continue;
            }
            
            cur_obj += sizeof(int);
            float NORM = *reinterpret_cast<float*>(cur_obj);
            cur_obj += sizeof(float);
            
            // Variables for filtering
            bool no_exact = false;
            float residual_NORM;
            float x = 0;
            bool is_left = true;
            float VAL = 0;
            
            // Create an array of pointers to table rows for this level
            std::vector<float*> table_ptrs_k(M2);

            // FILTER 1: Quick reject based on coarse distance
            if (std::abs(cell_dist_val) > cur_val) {
                x = std::abs(cell_dist_val) - cur_val;
                
                if (x >= delta_flag * NORM) {
                    continue;  // Skip this point
                } else if (points_examined > thres_pq && x >= delta * NORM) {
                    residual_NORM = NORM;
                    if (v >= 0) {
                        is_left = false;
                    }
                    cur_obj += offset00;
                    goto Label2;  // Skip to LSH filtering
                }
            }
            
            // FILTER 2: Multi-level refinement with PQ distances
            for (int k = 0; k < level; k++) {
                // Special handling for first level
                if (k == 0) {
                    if (points_examined <= thres_pq) {
                        // Use precomputed distance
                        VAL = *reinterpret_cast<float*>(cur_obj);
                        cur_obj += offset00;
                        residual_NORM = (*reinterpret_cast<float*>(cur_obj)) * NORM;
                        cur_obj += offset3;
                        goto Label;
                    } else {
                        cur_obj += offset00;
                        VAL = v;
                    }
                }
                
                // Compute accumulated distance approximation
                residual_NORM = (*reinterpret_cast<float*>(cur_obj)) * NORM;
                cur_obj += sizeof(float);
                
                // Add product quantization distance
                for (int i = 0; i < M2; i++) {
                    table_ptrs_k[i] = table2[k][i].data();
                }

                // Call pq_dist with the array of pointers
                VAL += NORM * pq_dist(reinterpret_cast<unsigned char*>(cur_obj), 
                                    table_ptrs_k.data(), M2);
                cur_obj += offset2;
                
Label:
                // Compute distance gap from current threshold
                if (VAL < 0) {
                    float ttmp = -VAL;
                    x = ttmp - cur_val;
                } else {
                    is_left = false;
                    x = VAL - cur_val;
                }
                
                // If gap is negative, point is promising
                if (x <= 0) {
                    break;
                } else {
                    // If gap exceeds residual norm, point cannot be good
                    if (x >= delta_flag * residual_NORM) {
                        no_exact = true;
                        break;
                    }
                    // If gap is close to threshold, use more aggressive filtering
                    else if (x >= delta * residual_NORM) {
                        break;
                    }
                }
            }
            
Label2:
            // LSH-based filtering for borderline cases
            if (!no_exact && x > 0) {
                cur_obj -= offset1;
                int collision_ = 0;
                
                // Count bit matches
                for (int jj = 0; jj < m_level; jj++) {
                    collision_ += fast_count(*reinterpret_cast<unsigned long*>(cur_obj), 
                                            query_proj[jj]);
                    cur_obj += sizeof(unsigned long);
                }
                
                // Use the quantile table and bit threshold computation
                int y_idx = static_cast<int>(cosine_inv * x / residual_NORM);
                if (y_idx >= cosine_inv)
                    y_idx = cosine_inv - 1;
                
                // Apply advanced bit threshold from precomputed table
                // This is more nuanced than the basic cosine table lookup
                float angle_ratio = static_cast<float>(y_idx) / cosine_inv;
                int lookup_idx = static_cast<int>(angle_ratio * max_inv);
                if (lookup_idx >= max_inv)
                    lookup_idx = max_inv - 1;
                
                // Get bit threshold from tab_inv using the lookup index
                int bit_threshold = tab_inv[lookup_idx];
                
                // Ensure we don't exceed the bounds
                int y = std::min(bit_threshold, cosine_table[y_idx]);
                
                // Check based on left/right side
                if (is_left) {
                    if (collision_ >= y) {
                        no_exact = true;
                    }
                } else {
                    collision_ = m - collision_;
                    if (collision_ >= y) {
                        no_exact = true;
                    }
                }
            }
            
            // Compute exact distance if passed all filters
            if (!no_exact) {
                float distance = compare_ip(data[point_id].data(), query.data(), dim) - u;
                if (distance < 0) {
                    distance = -distance;
                }
                
                visited_array[point_id] = visited_array_tag;
                
                // Skip if worse than current kth result
                if (distance >= retset[num < topk ? num - 1 : topk - 1].distance) {
                    continue;
                }
                
                // Add to results
                Neighbor nn;
                nn.id = point_id;
                nn.distance = distance;
                
                if (num < topk) {
                    InsertIntoPool(retset.data(), num, nn);
                    num++;
                } else {
                    InsertIntoPool(retset.data(), topk, nn);
                    cur_val = retset[topk - 1].distance;
                }
            }
        }
        
        // Ensure point counting is correct for early termination
        if (thres_flag && points_examined < thres_pq) {
            points_examined = thres_pq;
        }
    }
    
    // Release visited list
    visited_list_pool_->releaseVisitedList(vl);
    
    // Prepare final results
    std::vector<Neighbor> results;
    results.reserve(std::min(topk, num));
    
    for (int i = 0; i < std::min(topk, num); i++) {
        results.push_back(retset[i]);
    }
    
    return results;
}

#endif // MQH_H