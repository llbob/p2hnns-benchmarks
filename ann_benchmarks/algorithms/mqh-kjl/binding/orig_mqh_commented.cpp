#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <vector>
#include "mqh_lib/visited_list_pool.h"
#include <cmath>
#include <cassert>
#include <unordered_set>
#include <algorithm>
#include <limits.h>
#define L 256

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#define __builtin_popcount(t) __popcnt(t)
#else
#if defined(MQH_ARM)
// ARM-specific handling or fallback
#define NO_AVX
#define NO_SSE
#else
#ifdef HAVE_X86INTRIN
#include <x86intrin.h>
#else
#define NO_AVX
#define NO_SSE
#endif
#endif
#endif

using namespace std;
using namespace hnswlib;
class StopW
{
	std::chrono::steady_clock::time_point time_begin;

public:
	StopW()
	{
		time_begin = std::chrono::steady_clock::now();
	}

	float getElapsedTimeMicro()
	{
		std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
		return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
	}

	void reset()
	{
		time_begin = std::chrono::steady_clock::now();
	}
};
// This fast_count is used to count the number of bits that are different between two integers
static inline int fast_count(unsigned long a, unsigned long b)
{
	unsigned long u = a ^ b;
#ifdef _MSC_VER
	int count = __popcnt64(u);
#else
	int count = __builtin_popcountll(u);
#endif
	return count;
}
// this function is used to compare two float numbers
float compare_short(const float *a, const float *b, unsigned size)
{
	float dot0, dot1, dot2, dot3;
	const float *last = a + size;
	const float *unroll_group = last - 3;
	float result = 0;
	while (a < unroll_group)
	{
		dot0 = a[0] * b[0];
		dot1 = a[1] * b[1];
		dot2 = a[2] * b[2];
		dot3 = a[3] * b[3];
		result += dot0 + dot1 + dot2 + dot3;
		a += 4;
		b += 4;
	}
	while (a < last)
	{
		result += *a++ * *b++;
	}
	return result;
}
// The compare_ip function calculates the inner product (dot product) between two float vectors. It has a SIMD version, an AVX(even faster i think) version, and a SSE2 version and then a slower fallback version. The fallback version unrolls the loop to process 4 elements from each vector at a time and used direct multiplication and accumulation to calculate the dot product. It has a final loop to handle the remaining elements.
float compare_ip(const float *a, const float *b, unsigned size)
{
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
	if (DR)
	{
		AVX_DOT(e_l, e_r, sum, l0, r0);
	}

	for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
	{
		AVX_DOT(l, r, sum, l0, r0);
		AVX_DOT(l + 8, r + 8, sum, l1, r1);
	}
	_mm256_storeu_ps(unpack, sum);
	result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
	tmp1 = _mm128_loadu_ps(addr1);              \
	tmp2 = _mm128_loadu_ps(addr2);              \
	tmp1 = _mm128_mul_ps(tmp1, tmp2);           \
	dest = _mm128_add_ps(dest, tmp1);
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
	switch (DR)
	{
	case 12:
		SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
	case 8:
		SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
	case 4:
		SSE_DOT(e_l, e_r, sum, l0, r0);
	default:
		break;
	}
	for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
	{
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

	while (a < unroll_group)
	{
		dot0 = a[0] * b[0];
		dot1 = a[1] * b[1];
		dot2 = a[2] * b[2];
		dot3 = a[3] * b[3];
		result += dot0 + dot1 + dot2 + dot3;
		a += 4;
		b += 4;
	}
	while (a < last)
	{
		result += *a++ * *b++;
	}
#endif
#endif
#endif
	return result;
}

// Used for sorting - returns the difference between two float numbers
int comp_float(const void *a, const void *b)
{
	return *(float *)a - *(float *)b;
}
// Used for sorting - returns the difference between two integers
int comp_int(const void *a, const void *b)
{
	return *(int *)a - *(int *)b;
}

// This basic structure is for example used to store data points with their distances to quantization centroids, this is useful throughout the indexing as a way of assigning points to centroids and storing their distances, it's also useful for the residual distances.
struct elem
{
	int id;
	float val;
};

// This Q_elem represents a coarse queantization cell, it has two unsigned chars to represent the two quantization centroids that are used to represent the cell, and an integer to represent the number of points in the cell.
struct Q_elem
{
	unsigned char id1;
	unsigned char id2;
	int num;
};
// NOT IN USE: This seem to be an extended version of elem with an additional sorting value, allowing elements to be sorted based on a different metric than their primary value
struct elem2
{
	int id;
	float val;
	float sort_val;
};

// This Neighbor struct is used in the search function to store an ID and it's distance from a query point.
struct Neighbor
{
	int id;
	float distance;
};

// NOT IN USE: this is an extended version of Neighbor with an additional value to sort by, this seem to be used in the search function to store the ID distance and a value to sort by.
struct Neighbor2
{
	int id;
	float val;
	float sort_val;
};

// This is a function to compare two elem structs based on their val field, it's used in the qsort function in the ground_truth function to sort an array of elem structs.
int Elemcomp_a(const void *a, const void *b)
{
	elem x1 = *((elem *)b);
	elem x2 = *((elem *)a);

	if (x1.val > x2.val)
		return -1;
	else
	{
		return 1;
	}
}

// This is a function to compare two elem structs based on their val field, it's used in the qsort function in the index function to sort an array of elem structs by their val field.
int Elemcomp_d(const void *a, const void *b)
{
	elem x1 = *((elem *)b);
	elem x2 = *((elem *)a);

	if (x1.val > x2.val)
		return 1;
	else
	{
		return -1;
	}
}

// NOT IN USE: This function seems to be a third copy for elem2's
int Elemcomp2(const void *a, const void *b)
{

	elem2 x1 = *((elem2 *)b);
	elem2 x2 = *((elem2 *)a);

	if (x1.sort_val > x2.sort_val)
		return -1;
	else
	{
		return 1;
	}
}

// This function is used to generate a random float number between min and max, it's used in the gaussian function to generate random numbers.
float uniform(
	float min,
	float max)
{
	int num = rand();
	float base = (float)RAND_MAX - 1.0F;
	float frac = ((float)num) / base;

	return (max - min) * frac + min;
}

// This function is used to generate a random float number from a gaussian distribution with a given mean and sigma, it's used in the the indexing for LSH, meaning it's used to generate the random vectors with.
float gaussian(
	float mean,
	float sigma)
{
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

// This function seems to be for figuring out where to place an item in a sored array eg. a priqueue, but it's never used anywhere.
static inline int findinPool(elem *addr, int K, float val)
{
	int left = 0, right = K - 1;
	if (addr[left].val > val)
	{
		return left;
	}

	while (left < right - 1)
	{
		int mid = (left + right) / 2;
		if (addr[mid].val > val)
			right = mid;
		else
			left = mid;
	}
	return right;
}

// This function is used to maintain a fixed size priority queue, it's used in the search function to maintain a list of the K nearest neighbors to a query point, where each neighbor is represented by a Neighbor struct.
static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
{
	int left = 0, right = K - 1;
	if (addr[left].distance > nn.distance)
	{
		memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
		addr[left] = nn;
		return left;
	}
	if (addr[right].distance < nn.distance)
	{
		addr[K] = nn;
		return K;
	}
	while (left < right - 1)
	{
		int mid = (left + right) / 2;
		if (addr[mid].distance > nn.distance)
			right = mid;
		else
			left = mid;
	}

	while (left > 0)
	{
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

// NOT IN USE: This function is used to maintain a fixed size priority queue, it's used in the search function to maintain a list of the K nearest neighbors to a query point, where each neighbor is represented by a Neighbor2 struct.
static inline int InsertIntoPool2(Neighbor2 *addr, unsigned K, Neighbor2 nn)
{
	int left = 0, right = K - 1;
	if (addr[left].sort_val > nn.sort_val)
	{
		memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
		addr[left] = nn;
		return left;
	}

	if (addr[right].sort_val < nn.sort_val)
	{
		addr[K] = nn;
		return K;
	}
	while (left < right - 1)
	{
		int mid = (left + right) / 2;
		if (addr[mid].sort_val > nn.sort_val)
			right = mid;
		else
			left = mid;
	}

	while (left > 0)
	{
		if (addr[left].sort_val < nn.sort_val)
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

// This is the kmeans algorithm used for mqh, it's run 2 times during coarse quantization and run M2 times per level during norm explicit residual quantization. Its initialized to L size which is globally defined here to be 256, meaning we'll use 256 centroids for the quantization. Eg. in coarse quantization we will use the Q_elem struct to keep track of these combinations of cells and their number of connected points.
void K_means(float **train, double **vec, int n_sample, int d)
{

	int seed_ = 1;
	int cur_obj = 0;
	int *array_ = new int[L];
	bool flag_ = false;

	for (int i = 0; i < L; i++)
	{
		srand(seed_);
		seed_++;
		int l = rand() % n_sample;
		for (int j = 0; j < d; j++)
		{
			vec[i][j] = train[l][j];
		}
		flag_ = false;

		for (int j = 0; j < cur_obj; j++)
		{
			if (l == array_[j])
			{
				i--;
				flag_ = true;
				break;
			}
		}
		if (flag_ == false)
		{
			array_[cur_obj] = l;
			cur_obj++;
		}
	}

	delete[] array_;

	float sum, min_sum;
	int vec_id;
	int *pvec = new int[n_sample];
	int *count = new int[L];
	int ROUND = 20;

	for (int k = 0; k < ROUND; k++)
	{
		for (int j = 0; j < L; j++)
			count[j] = 0;

		for (int j = 0; j < n_sample; j++)
		{
			for (int l = 0; l < L; l++)
			{
				sum = 0;
				for (int i = 0; i < d; i++)
				{
					sum += (train[j][i] - vec[l][i]) * (train[j][i] - vec[l][i]);
				}
				if (l == 0)
				{
					min_sum = sum;
					vec_id = 0;
				}
				else if (sum < min_sum)
				{
					min_sum = sum;
					vec_id = l;
				}
			}
			pvec[j] = vec_id;
			count[pvec[j]]++;
		}

		for (int j = 0; j < n_sample; j++)
		{
			for (int i = 0; i < d; i++)
			{
				vec[pvec[j]][i] = 0;
			}
		}

		for (int j = 0; j < n_sample; j++)
		{
			for (int i = 0; i < d; i++)
			{
				vec[pvec[j]][i] += train[j][i];
			}
		}

		for (int j = 0; j < L; j++)
		{
			if (count[j] == 0)
				continue;
			for (int i = 0; i < d; i++)
			{
				vec[j][i] = vec[j][i] / count[j];
			}
		}
	}
	delete[] count;
	delete[] pvec;
}

// This function is a basic one used to calculate the norm of a float array, it's used in the indexing to calculate the norm of the residual vectors.
float calc_norm(float *array, int d)
{
	float sum = 0;
	for (int i = 0; i < d; i++)
	{
		sum += array[i] * array[i];
	}
	return sqrt(sum);
}
// The select_sample function systematically selects a subset of data points from a larger dataset using evenly spaced sampling this is to avoid training on the full dataset. it's used for all kmeans, it effectively is a slimmed version of the data set with variable name train..
void select_sample(float **data, float **train, int n_pts, int size, int dim)
{
	int interval = n_pts / size;

	int cur_obj = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			train[i][j] = data[cur_obj][j];
		}
		cur_obj += interval;
	}
}

// This function is meant to efficiently compute approximate distances.  For each candidate point, it uses the point's quantization codes to quickly look up and sum the distances
// This shoudl avoid the more expensive vector operations with the full-dimensional data.
// The line in search: float distance = cur_dist + z * y; combines the coarse quantization distance with the product quantization distance
static inline float pq_dist(unsigned char *a, float **b, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++)
	{
		unsigned char x = a[i];
		sum += b[i][x];
	}
	return sum;
}

void ground_truth(int n_pts, int d, int n_query, char *path_data, char *query_data, char *path_gt)
{
	int maxk = 100;
	float **data = new float *[n_pts];
	for (int i = 0; i < n_pts; i++)
		data[i] = new float[d];

	ifstream input(path_data, ios::binary);
	for (int i = 0; i < n_pts; i++)
	{
		int t;
		input.read((char *)&t, 4);
		input.read((char *)(data[i]), sizeof(float) * d);
	}

	input.close();
	float **query = new float *[n_query];
	float *u = new float[n_query];

	for (int i = 0; i < n_query; i++)
		query[i] = new float[d];

	ifstream input_query(query_data, ios::binary);
	for (int i = 0; i < n_query; i++)
	{
		int t;
		input_query.read((char *)&t, 4);
		input_query.read((char *)(query[i]), 4 * d);
		input_query.read((char *)&(u[i]), 4);
	}

	input_query.close();

	ofstream outputGT(path_gt, ios::binary);

	elem *results = new elem[n_pts];
	for (int i = 0; i < n_query; i++)
	{
		for (int j = 0; j < n_pts; j++)
		{
			float distance = compare_ip(data[j], query[i], d) - u[i];
			if (distance < 0)
				distance = -1 * distance;

			results[j].id = j;
			results[j].val = distance;
		}
		qsort(results, n_pts, sizeof(elem), Elemcomp_a);

		outputGT.write((char *)&i, sizeof(int));
		for (int j = 0; j < maxk; j++)
		{
			outputGT.write((char *)(results[j].id), sizeof(int));
		}
	}
	outputGT.close();
}

// This is where we build the MQH index
void index(int n_pts, int d, int n_sample, char *path_data, char *index_data)
{
	int M2 = 16;		   // Big M, number of subcodebooks, meaning number of kmeans runs
	int level = 4;		   // total number of levels
	int size = M2 * level; // total number of subcodebooks across levels

	int m_level = 1;		 // number of hash tables per level
	int m_num = 64;			 // number of projection vectors per hash table, amount of bits
	int m = m_level * m_num; // total number of hash functions per level? or in total?

	int d_org = d;	 // original dimension of data points
	int d_supp;		 // variable for padding to add to dimensions
	if (d % M2 == 0) // if d is divisible by M2, then no padding is needed
		d_supp = 0;
	else
	{
		d_supp = M2 - d % M2;
	} // if not, then we need to add padding to make it divisible by M2
	d = d + d_supp;	 // new dimension of data points
	int d2 = d / M2; // dimension of subcodebooks

	// Allocate memory for the dataset (with padded dimensions if necessary)
	float **data = new float *[n_pts];
	for (int i = 0; i < n_pts; i++)
		data[i] = new float[d];
	// Initialize all vector elements to 0, (padded dimensions will stay 0)
	for (int i = 0; i < n_pts; i++)
		for (int j = 0; j < d; j++)
			data[i][j] = 0;
	// Read the data from the binary file into the data array
	ifstream input(path_data, ios::binary);
	for (int i = 0; i < n_pts; i++)
	{
		float t;
		input.read((char *)(data[i]), sizeof(float) * d_org);
		input.read((char *)&t, sizeof(float)); // read additional dimension
		if ((int)t != 1)
			std::cout << "dimensin error" << std::endl;
	}
	// open the output file for writing the index structure
	ofstream output(index_data, ios::binary);

	input.close();
	// Start timer to measure indexing time
	StopW stopw = StopW();

	// Variables for distance calculations and finding minimum distances
	float sum = 0;
	float min_sum;
	int min_id;
	// Allocate memory for the norm of each data point
	float *norm = new float[n_pts];
	// Allocate memory for residual vectors (used after coarse quantization)
	float **residual_vec = new float *[n_pts];
	for (int i = 0; i < n_pts; i++)
	{
		residual_vec[i] = new float[d];
	}
	// Allocate memory for the first set of coarse quantizer centroids (first half of dimensions)
	double **vec_1 = new double *[L];
	for (int i = 0; i < L; i++)
		vec_1[i] = new double[d / 2];
	// Allocate memory for the second set of coarse quantizer centroids (second half of dimensions)
	double **vec_2 = new double *[L];
	for (int i = 0; i < L; i++)
		vec_2[i] = new double[d / 2];
	// Allocate memory for training samples
	float **train = new float *[n_sample];
	for (int i = 0; i < n_sample; i++)
		train[i] = new float[d];
	// Calculate the norm (magnitude) of each data point
	for (int i = 0; i < n_pts; i++)
	{
		norm[i] = calc_norm(data[i], d);
	}
	// Select a representative subset of data for training quantizers
	select_sample(data, train, n_pts, n_sample, d);
	// Split training data into two halves for the two separate quantizers
	float **train1 = new float *[n_sample];
	for (int i = 0; i < n_sample; i++)
		train1[i] = new float[d / 2];

	float **train2 = new float *[n_sample];
	for (int i = 0; i < n_sample; i++)
		train2[i] = new float[d / 2];
	// divide each training vector into first and second halves
	for (int i = 0; i < n_sample; i++)
	{
		// first half of dimensions
		for (int j = 0; j < d / 2; j++)
		{
			train1[i][j] = train[i][j];
		}
		// second half of dimensions
		for (int j = 0; j < d / 2; j++)
		{
			train2[i][j] = train[i][j + d / 2];
		}
	}
	// run K-means clustering on the first half of each training vector to create first codebook
	K_means(train1, vec_1, n_sample, d / 2);
	// run K-means clustering on the second half of each training vector to create second codebook
	K_means(train2, vec_2, n_sample, d / 2);
	// array to count how many data points fall into each combination of clusters
	int *count = new int[L * L];
	unsigned char *vec_id1 = new unsigned char[n_pts];
	unsigned char *vec_id2 = new unsigned char[n_pts];
	// onitialize all counts to zero
	for (int i = 0; i < L * L; i++)
		count[i] = 0;
	// for each data point, find its closest centroid in each half of the dimensions
	for (int i = 0; i < n_pts; i++)
	{
		// find closest centroid for first half of dimensions
		for (int j = 0; j < L; j++)
		{
			sum = 0;
			for (int l = 0; l < d / 2; l++)
			{
				sum += (data[i][l] - vec_1[j][l]) * (data[i][l] - vec_1[j][l]);
			}

			if (j == 0)
			{
				min_sum = sum;
				min_id = 0;
			}
			else
			{
				if (sum < min_sum)
				{
					min_sum = sum;
					min_id = j;
				}
			}
		}
		vec_id1[i] = min_id; // store closest centroid ID for first half

		// find closest centroid for second half of dimensions
		for (int j = 0; j < L; j++)
		{
			sum = 0;
			for (int l = 0; l < d / 2; l++)
			{
				sum += (data[i][l + d / 2] - vec_2[j][l]) * (data[i][l + d / 2] - vec_2[j][l]);
			}

			if (j == 0)
			{
				min_sum = sum;
				min_id = 0;
			}
			else
			{
				if (sum < min_sum)
				{
					min_sum = sum;
					min_id = j;
				}
			}
		}
		vec_id2[i] = min_id; // store closest centroid ID for second half
		// incremen count for this combination of centroids
		count[vec_id1[i] * L + vec_id2[i]]++;
	}
	// write first set of centroids to index file
	for (int i = 0; i < L; i++)
	{
		for (int j = 0; j < d / 2; j++)
		{
			float x = vec_1[i][j];
			output.write((char *)&x, sizeof(float));
		}
	}
	// write second set of centroids to index file
	for (int i = 0; i < L; i++)
	{
		for (int j = 0; j < d / 2; j++)
		{
			float x = vec_2[i][j];
			output.write((char *)&x, sizeof(float));
		}
	}
	// count how many distinct cluster combinations are actually used
	int n_cand1 = 0;
	int *map_table = new int[L * L];
	for (int i = 0; i < L * L; i++)
	{
		map_table[i] = -1; // initialize mapping table entries to -1
		if (count[i] > 0)
			n_cand1++; // count only combinations with at least one point
	}
}
// allocate memory for the array of elements for each cluster combination basically of size L*L minus the unused ones
elem **array1 = new elem *[n_cand1];
int *n_temp = new int[n_cand1]; // alloc for counter for populating each array
for (int i = 0; i < n_cand1; i++)
	n_temp[i] = 0;
Q_elem *pq_M2 = new Q_elem[n_cand1]; // allocate memory to store cluster information for each used combination

// Build mapping from original L×L matrix to compact representation
n_cand1 = 0;
for (int i = 0; i < L * L; i++)
{
	if (count[i] > 0)
	{
		array1[n_cand1] = new elem[count[i]]; // Allocate memory for points in this cluster combination
		map_table[i] = n_cand1;				  // Map original index to compact index
		// Store centroid IDs and count for this cluster
		pq_M2[n_cand1].id1 = i / L;	   // first half centroid ID
		pq_M2[n_cand1].id2 = i % L;	   // second half centroid ID
		pq_M2[n_cand1].num = count[i]; // number of points in this cluster

		n_cand1++;
	}
}
// Write quantization information to index file
output.write((char *)&(n_cand1), 4);
for (int i = 0; i < n_cand1; i++)
{ // Number of used cluster combinations
	output.write((char *)&(pq_M2[i].id1), sizeof(unsigned char));
	output.write((char *)&(pq_M2[i].id2), sizeof(unsigned char));
	output.write((char *)&(pq_M2[i].num), sizeof(int));
}
// Create compact array of counts for used clusters
int s = 0;
int *count1 = new int[n_cand1];
for (int i = 0; i < L * L; i++)
{
	if (count[i] > 0)
	{
		count1[s] = count[i];
		s++;
	}
}
// Allocate memory for reconstructing quantized vectors
float *vec = new float[d];

float ttest2;
// For each data point, compute and store residual vectors
for (int i = 0; i < n_pts; i++)
{
	// Find which cluster this point belongs to
	int temp = vec_id1[i] * L + vec_id2[i];
	int table_id = map_table[temp];
	array1[table_id][n_temp[table_id]].id = i;

	// Reconstruct the quantized approximation from the two centroids
	for (int j = 0; j < d / 2; j++)
	{
		vec[j] = vec_1[vec_id1[i]][j]; // First half of d
	}

	for (int j = 0; j < d / 2; j++)
	{
		vec[j + d / 2] = vec_2[vec_id2[i]][j]; // Second half of d
	}

	// Compute residual vector (difference between original and quantized)
	for (int j = 0; j < d; j++)
	{
		residual_vec[i][j] = data[i][j] - vec[j];
	}

	// Calculate L2 norm of the residual vector
	array1[table_id][n_temp[table_id]].val = 0;
	for (int j = 0; j < d; j++)
		array1[table_id][n_temp[table_id]].val += residual_vec[i][j] * residual_vec[i][j];

	array1[table_id][n_temp[table_id]].val = sqrt(array1[table_id][n_temp[table_id]].val);
	n_temp[table_id]++;
}

// Sort points in each cluster combination by residual distance
for (int i = 0; i < n_cand1; i++)
{
	qsort(array1[i], count1[i], sizeof(elem), Elemcomp_d);
}
// Write sorted points and their residual norms to the index file
for (int i = 0; i < n_cand1; i++)
{
	for (int j = 0; j < count1[i]; j++)
	{
		output.write((char *)&(array1[i][j].id), sizeof(int));
		output.write((char *)&(array1[i][j].val), sizeof(float));
	}
}
// Allocate memory for calculating norms
float *norm2 = new float[n_pts];
// Calculate the norm of each residual vector
for (int i = 0; i < n_pts; i++)
{
	norm2[i] = calc_norm(residual_vec[i], d);
}

// A flag to track which vectors have a nearzero norm
bool *zero_flag = new bool[n_pts];
for (int i = 0; i < n_pts; i++)
{
	zero_flag[i] = false;
}
// Minimum value threshold for flag, point is to avoid division by zero
float min_float = 0.0000001;

// Process each data point's residual vector
for (int i = 0; i < n_pts; i++)
{
	// Check if residual vector has near-zero magnitude
	if (norm2[i] < min_float)
	{
		zero_flag[i] == true; // Bug: should be = not ==??
							  // Replace near-zero residuals with unit vector along first dimension
		residual_vec[i][0] = 1;
		for (int j = 1; j < d; j++)
		{
			residual_vec[i][j] = 0;
		}
	}
	else
	{
		// Normalize non-zero residual vectors to unit length
		// This separates magnitude information from directional information NERQ
		for (int j = 0; j < d; j++)
		{
			residual_vec[i][j] = residual_vec[i][j] / norm2[i];
		}
	}
}
// Allocate memory for all centroids for multilevel quantization
double ***vec2 = new double **[size];
for (int i = 0; i < size; i++)
	vec2[i] = new double *[L];
for (int i = 0; i < size; i++)
{
	for (int j = 0; j < L; j++)
	{
		vec2[i][j] = new double[d2];
	}
}
// Temporary storage for training samples for product quantization
float **residual_pq = new float *[n_sample];
for (int i = 0; i < n_sample; i++)
{
	residual_pq[i] = new float[d2];
}
// Storage for product quantization codes for each data point
unsigned char **pq_id = new unsigned char *[n_pts];
for (int i = 0; i < n_pts; i++)
{
	pq_id[i] = new unsigned char[M2];
}
// Generate random projection vectors for locality-sensitive hashing (LSH)
float **proj_array = new float *[m];
for (int i = 0; i < m; ++i)
{
	proj_array[i] = new float[d];
	for (int j = 0; j < d; ++j)
	{
		proj_array[i][j] = gaussian(0.0f, 1.0f);
	}
}
// Write projection vectors to index file
for (int i = 0; i < m; ++i)
	output.write((char *)(proj_array[i]), sizeof(float) * d);
// Compute binary hash codes for normalized residual vectors
for (int i = 0; i < n_pts; i++)
{
	for (int j = 0; j < m_level; j++)
	{
		unsigned long code_num = 0;
		for (int l = 0; l < m_num; l++)
		{
			float ssum = 0;
			for (int ll = 0; ll < d; ll++)
				ssum += residual_vec[i][ll] * proj_array[j * m_num + l][ll];

			if (ssum >= 0)
			{
				code_num += 1;
			}

			if (l < m_num - 1)
				code_num = code_num << 1;
		}
		output.write((char *)&code_num, sizeof(unsigned long));
	}
}
// Allocate temporary storage for projection values
float *proj_temp = new float[m];
float **proj_val = new float *[n_pts];
for (int i = 0; i < n_pts; i++)
	proj_val[i] = new float[m];

// Begin multilevel product quantization
float *test_vec = new float[d];
for (int k = 0; k < level; k++)
{
	// For each subspace, train quantizers on residual subvectors
	for (int i = 0; i < M2; i++)
	{
		int ccount = 0;
		// Collect training samples for this subspace
		for (int j = 0; j < n_pts; j++)
		{
			if (zero_flag[j] == true)
			{
				continue; // Skip zero vectors
			}
			// Extract subvector for current subspace
			for (int l = 0; l < d2; l++)
			{
				residual_pq[ccount][l] = residual_vec[j][i * d2 + l];
			}
			ccount++;
			if (ccount >= n_sample)
				break;
		}
		// Run K-means to create codebook for this subspace
		K_means(residual_pq, vec2[k * M2 + i], n_sample, d2);
	}

	// Second step: Assign each data point to closest centroid in each subspace
	for (int n = 0; n < n_pts; n++)
	{
		for (int i = 0; i < M2; i++)
		{
			// Find closest centroid from the 256 options
			for (int j = 0; j < L; j++)
			{
				sum = 0;
				// Calculate squared Euclidean distance
				for (int l = 0; l < d2; l++)
				{
					sum += (residual_vec[n][i * d2 + l] - vec2[k * M2 + i][j][l]) * (residual_vec[n][i * d2 + l] - vec2[k * M2 + i][j][l]);
				}
				// Keep track of minimum distance and corresponding centroid
				if (j == 0)
				{
					min_sum = sum;
					min_id = 0;
				}
				else
				{
					if (sum < min_sum)
					{
						min_sum = sum;
						min_id = j;
					}
				}
			}
			// Store the closest centroid ID for this subspace
			pq_id[n][i] = min_id;
		}
	}
	// Third step: Compute new residuals by subtracting quantized vectors
	for (int n = 0; n < n_pts; n++)
	{
		for (int j = 0; j < M2; j++)
		{
			int temp_M = k * M2 + j;
			for (int l = 0; l < d2; l++)
			{
				// Calculate new residual = old residual - chosen centroid
				residual_vec[n][j * d2 + l] = residual_vec[n][j * d2 + l] - vec2[temp_M][pq_id[n][j]][l];
				test_vec[j * d2 + l] = vec2[temp_M][pq_id[n][j]][l];
			}
		}
		// Calculate norm of new residual vector
		float sum = 0;
		for (int j = 0; j < d; j++)
		{
			sum += residual_vec[n][j] * residual_vec[n][j];
		}
		norm2[n] = sqrt(sum);
		// Flag vectors with negligible residuals
		if (norm2[n] < min_float)
			zero_flag[n] = true;
	}
	// Write quantization data for this level to the index file
	for (int i = 0; i < n_pts; i++)
	{
		output.write((char *)&(norm2[i]), sizeof(float));
		output.write((char *)(pq_id[i]), M2);
	}
	//This very important point we aremissing
	// Generate and write LSH codes for residual vectors at this level
	for (int i = 0; i < n_pts; i++)
	{
		for (int j = 0; j < m_level; j++)
		{
			unsigned long code_num = 0;
			// Generate 64-bit hash code using random projections
			for (int l = 0; l < m_num; l++)
			{
				float ssum = 0;
				// Compute dot product with random projection vector
				for (int ll = 0; ll < d; ll++)
					ssum += residual_vec[i][ll] * proj_array[j * m_num + l][ll];
				// Set bit if projection is positive
				if (ssum >= 0)
				{
					code_num += 1;
				}
				// Shift bits for next projection
				if (l < m_num - 1)
					code_num = code_num << 1;
			}
			output.write((char *)&code_num, sizeof(unsigned long));
		}
	}
}
// Write all product quantization codebooks to the index file
for (int i = 0; i < size; i++)
{
	for (int j = 0; j < L; j++)
	{
		for (int l = 0; l < d2; l++)
		{
			float x = vec2[i][j][l];
			output.write((char *)&x, sizeof(float));
		}
	}
}

output.close();
// Report total indexing time
float time_us_indexing = stopw.getElapsedTimeMicro();
cout << time_us_indexing / 1000 / 1000 << " s" << "\n";
}

float Quantile(float *table, float a, int size)
{
	int i = 0;
	for (i = 0; i < size; i++)
	{
		if (a < table[i])
			break;
	}
	return (1.0f * i / 100);
}

void search(int n_pts, int n_query, int d, int topk, float delta, int l0, int flag_, char *query_data, char *base_data, char *path_gt, char *path_index)
{
	int real_topk = topk;
	topk = 100;

	int maxk = 100;
	int thres_pq = n_pts / 10;
	int thres_pq2 = n_pts / 2;

	int n_exact = 2000;

	std::vector<Neighbor> retset(topk + 1);
	std::vector<Neighbor> retset2(n_exact + 1);
	int M2 = 16; // num subcodebooks
	int level = 4;
	int size = M2 * level;
	int m_level = 1;
	int m_num = 64; // num of random projection vectors

	int m = m_level * m_num;

	float delta1;
	if (flag_ == 1) // For guarantees on recall rate - use maximum filtering
		delta1 = 1;
	else
	{
		delta1 = delta; // for approx. nearest neighbors - use provided delta
	}

	int offset0 = l0;

	int m2 = m;
	float half_m2 = m2 / 2; // NOT IN USE: don't know the point of it

	// Used to track visited points - this is a pq
	VisitedListPool *visited_list_pool_ = new VisitedListPool(1, n_pts);

	int d_org = d;
	int d_supp; // Padding again, as in the indexing part
	if (d % M2 == 0)
		d_supp = 0;
	else
	{
		d_supp = M2 - d % M2;
	}
	d = d + d_supp;
	int d2 = d / M2;

	float PI = 3.1415926535;
	// Cosine lookup table for angle-based filtering
	int cosine_inv = 100;
	int *cosine_table = new int[cosine_inv];

	for (int i = 0; i < cosine_inv; i++)
	{
		// Convert index i to cosine value (0 to 0.99) and calculate corresponding angle
    	// Then scale to hash bit space (m bits) and add base offset
		cosine_table[i] = m * acos(1.0f * i / cosine_inv) / PI + offset0;
		// Ensure the threshold doesn't exceed the total number of hash bits
		if (cosine_table[i] > m)
			cosine_table[i] = m;
	}

	// allocate memory for random projection vectors
	float **proj_array = new float *[m];
	for (int i = 0; i < m; ++i)
		proj_array[i] = new float[d];

	// Constants for probabilistic search guarantees
	float epsilon = 0.99999; // Desired success probability (very close to 1)
	float alpha = 0.673; // LSH parameter for controlling collision probability


	// Calculate temperature parameter for bit sampling based on probability theory
	// This determines how many bits need to match for candidates to be considered
	float temp = sqrt(log(1 / epsilon) / 2 / m);


	// Create table of quantile values for the normal distribution
	// Used to map angular distances to bit count thresholds with statistical guarantees
	int table_size = 170;
	float *quantile_table = new float[table_size];
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

	// Calculate the statistical threshold for query-to-hash comparison based on probability theory

	float coeff = Quantile(quantile_table, 0.5 * temp + 0.75, table_size);
	
	// Recalculate temperature parameter to balance recall and precision
	temp = sqrt(log(1 / epsilon) / 2 / m);
	// Calculate lower bound coefficient for bit matching threshold
	float coeff2 = Quantile(quantile_table, 0.5 * temp + 0.75, table_size);
	// Calculate upper bound coefficient for maximum allowable bit differences
	float max_coeff = Quantile(quantile_table, 0.5 * (1 - temp) + 0.5, table_size);

	// Create a lookup table for fast mapping between distance ratios and bit thresholds
	int max_inv = 1000; // Number of discretized intervals
	unsigned char *tab_inv = new unsigned char[max_inv];

	// Calculate step size for interpolating between coefficients
	float ratio = (max_coeff - coeff2) / max_inv;
	for (int i = 0; i < max_inv; i++)
	{
		// Interpolate between min and max coefficients
		float temp2 = coeff2 + i * ratio;

		// Map to closest entry in quantile table
		int temp3 = temp2 * 100;
		if (temp3 >= table_size)
			temp3 = table_size - 1;

    	// Convert quantile value to Hamming distance threshold
    	// This transforms probability space into bit-count space
		temp2 = 2 * (quantile_table[temp3] - 0.5) + temp;

		// Convert to actual bit count and ensure it doesn't exceed total bits
		tab_inv[i] = temp2 * m + 1;
		if (tab_inv[i] > m)
			tab_inv[i] = m;
	}

	// Create another lookup table for direct quantile-to-bits conversion
	float *ttab = new float[table_size];
	for (int i = 0; i < table_size; i++)
	{
		// Convert normalized quantile values to bit differences
    	// Centered at 0.5 and scaled to match hash code size
		ttab[i] = (quantile_table[i] - 0.5) * 2 * m;
	}

	// Allocate memory for ground truth results (used for validation)
	int *massQA = new int[n_query * maxk];

	// Open ground truth file
	FILE *fp = fopen(path_gt, "r");
	if (!fp)
	{
		printf("Could not open %s\n", path_gt);
		exit(0);
	}

	// Verify ground truth file format matches expected dimensions
	int tmp1 = -1, tmp2 = -1;
	fscanf(fp, "%d,%d\n", &tmp1, &tmp2);
	assert(tmp1 == n_query && tmp2 == maxk);

	// Read ground truth data for each query
	for (int i = 0; i < n_query; ++i)
	{
		fscanf(fp, "%d", &tmp1);
		for (int j = 0; j < maxk; ++j)
		{
			float tmp3;
			fscanf(fp, ",%d,%f", massQA + maxk * i + j, &tmp3);
		}
		fscanf(fp, "\n");
	}
	fclose(fp);

	// Allocate memory for search results
	int **search_result = new int *[n_query];
	for (int i = 0; i < n_query; i++)
		search_result[i] = new int[topk];

	// Allocate memory for dataset points
	float **data = new float *[n_pts];
	for (int i = 0; i < n_pts; i++)
		data[i] = new float[d];

	// Init all to 0
	for (int i = 0; i < n_pts; i++)
		for (int j = 0; j < d; j++)
			data[i][j] = 0;

	// Load dataset from bin file
	ifstream input_data(base_data, ios::binary);
	for (int i = 0; i < n_pts; i++)
	{
		float t;
		input_data.read((char *)(data[i]), 4 * d_org);
		input_data.read((char *)&t, sizeof(float));
	}

	input_data.close();

	// Allocate memory for query vectors and inner product thresholds
	float **query = new float *[n_query];
	float *u = new float[n_query];

	for (int i = 0; i < n_query; i++)
		query[i] = new float[d];

	// Initialize query vectors to zero
	for (int i = 0; i < n_query; i++)
		for (int j = 0; j < d; j++)
			query[i][j] = 0;

	// Load query vectors from binary file
	ifstream input_query(query_data, ios::binary);
	for (int i = 0; i < n_query; i++)
	{
		// Read query vector
		input_query.read((char *)(query[i]), sizeof(float) * d_org);
		// Normalize query vector to unit length
		float norm_query = calc_norm(query[i], d);

		for (int j = 0; j < d; j++)
			query[i][j] = query[i][j] / norm_query;

		// Read inner product threshold and scale by query norm
		input_query.read((char *)&(u[i]), sizeof(float));

		u[i] = -1 * u[i] / norm_query;
	}

	input_query.close();

	// Open the precomputed index file
	ifstream input_index(path_index, ios::binary);

	// Alloc mem for first subspace from coarse quantization
	float **vec1 = new float *[L];
	for (int i = 0; i < L; i++)
		vec1[i] = new float[d / 2];

	// Alloc mem for second subspace from coarse quantization
	float **vec2 = new float *[L];
	for (int i = 0; i < L; i++)
		vec2[i] = new float[d / 2];

	// Read first set of coarse quantization centroids
	for (int i = 0; i < L; i++)
	{
		for (int j = 0; j < d / 2; j++)
		{
			float t;
			input_index.read((char *)&t, sizeof(float));
			vec1[i][j] = t;
		}
	}


	// read second set of coarse quantization centroids
	for (int i = 0; i < L; i++)
	{
		for (int j = 0; j < d / 2; j++)
		{
			float t;
			input_index.read((char *)&t, sizeof(float));
			vec2[i][j] = t;
		}
	}

	// Read the number of non-empty cells in the coarse quantization
	int n_cand1;
	input_index.read((char *)&(n_cand1), 4);

	// Allocate memory for coarse quantization data
	Q_elem *pq_M2 = new Q_elem[n_cand1];
	// Array to store the number of points in each cell
	int *count = new int[n_cand1];
	for (int i = 0; i < n_cand1; i++)
	{
		// Read the first centroid ID (first half)
		input_index.read((char *)&(pq_M2[i].id1), 1);
		// Read the second centroid ID (second half)
		input_index.read((char *)&(pq_M2[i].id2), 1);
		// Read the number of points in this cell
		input_index.read((char *)&(count[i]), sizeof(int));
		pq_M2[i].num = count[i];
	}

	// alloc memory for storing point IDs for each cluster
	int **coarse_index = new int *[n_cand1];
	for (int i = 0; i < n_cand1; i++)
		coarse_index[i] = new int[count[i]];

	// Allocate memory for the packed index structure
	// This will store all information for each data point in a compact layout
	char **index_ = (char **)malloc(sizeof(void *) * n_cand1);

	// Calculate memory layout sizes for the index structure
	// Each data point requires space for:
	// - point ID (int)
	// - residual norm (float)
	// - additional metadata (float)
	// - LSH codes (unsigned long * m_level)
	// - For each level: PQ codes, norm, and LSH codes
	int size_per_element_ = sizeof(int) + 2 * sizeof(float) + sizeof(unsigned long) * m_level + level * (M2 + sizeof(float) + sizeof(unsigned long) * m_level);
	int proj_per_element_ = level * m * sizeof(float);
	
	// For each cluster, allocate and populate its portion of the index
	for (int i = 0; i < n_cand1; i++)
	{
		// Allocate memory for all points in this cluster
		index_[i] = (char *)malloc(count[i] * size_per_element_);

		char *cur_loc = index_[i];

		// Read point IDs and residual norms for this cluster
		for (int j = 0; j < count[i]; j++)
		{
			float b; // Residual norm
			input_index.read((char *)&(coarse_index[i][j]), 4); // read point ID
			input_index.read((char *)&(b), 4); // read residual norm

			// Store point ID at the beginning of this point's data block
			memcpy(cur_loc, &(coarse_index[i][j]), sizeof(int));
			cur_loc += sizeof(int);
			// Store residual norm
			memcpy(cur_loc, &b, sizeof(float));
			// Skip to the next point's data block
			cur_loc += (size_per_element_ - sizeof(int));
		}
	}
	
	// Read rand projection vectors
	for (int i = 0; i < m; i++)
		input_index.read((char *)(proj_array[i]), sizeof(float) * d);

	// Alloc memory for storing hash codes of all data points
	unsigned long **rough_code = new unsigned long *[n_pts];
	for (int i = 0; i < n_pts; i++)
		rough_code[i] = new unsigned long[m_level];

	// Read hash codes for all data points
	for (int i = 0; i < n_pts; i++)
	{
		for (int j = 0; j < m_level; j++)
		{
			input_index.read((char *)&(rough_code[i][j]), sizeof(unsigned long));
		}
	}

	// Copy hash codes to the compact index structure by cluster
	for (int i = 0; i < n_cand1; i++)
	{
		char *cur_loc = index_[i];
		for (int j = 0; j < count[i]; j++)
		{
			// Position pointer after point ID and residual norm
			cur_loc = index_[i] + j * size_per_element_ + sizeof(int) + 2 * sizeof(float);

			// Get point ID from coarse index
			int b = coarse_index[i][j];

			// Copy hash codes for this point
			for (int jj = 0; jj < m_level; jj++)
			{
				memcpy(cur_loc, &(rough_code[b][jj]), sizeof(unsigned long));
				cur_loc += sizeof(unsigned long);
			}
		}
	}

	// Allocate memory for product quantization codes for all points
	unsigned char **pq_id = new unsigned char *[n_pts];
	for (int i = 0; i < n_pts; i++)
	{
		pq_id[i] = new unsigned char[M2];
	}
	// Allocate memory for residual norms
	float *residual_norm = new float[n_pts];

	// Allocate memory for residual hash codes
	unsigned long **code_num = new unsigned long *[n_pts];
	for (int i = 0; i < n_pts; i++)
		code_num[i] = new unsigned long[m_level];

	// For each refinement level, load and organize the data
	for (int i = 0; i < level; i++)
	{
		// Read residual norms and PQ codes for all points at this level
		for (int j = 0; j < n_pts; j++)
		{
			input_index.read((char *)&(residual_norm[j]), sizeof(float));
			input_index.read((char *)(pq_id[j]), M2);
		}
		
		// Copy residual norms and PQ codes to the compact index structure by cluster
		for (int k = 0; k < n_cand1; k++)
		{
			for (int j = 0; j < count[k]; j++)
			{
				int a = coarse_index[k][j]; // Get point ID
				// Position pointer at this levels data for this point
				char *cur_loc = index_[k] + j * size_per_element_ + sizeof(int) + 2 * sizeof(float) + sizeof(unsigned long) * m_level + i * (M2 + sizeof(float) + sizeof(unsigned long) * m_level);

				// Copy residual norm 
				memcpy(cur_loc, &(residual_norm[a]), sizeof(float));
				cur_loc += sizeof(float);
				// and PQ codes for this point
				for (int l = 0; l < M2; l++)
				{
					memcpy(cur_loc, &(pq_id[a][l]), 1);
					cur_loc += 1;
				}
			}
		}

		// Read hash codes for residual vectors at this level
		for (int j = 0; j < n_pts; j++)
		{
			for (int jj = 0; jj < m_level; jj++)
				input_index.read((char *)&(code_num[j][jj]), sizeof(unsigned long));
		}

		// Copy residual hash codes to the compact index structure by cluster
		for (int k = 0; k < n_cand1; k++)
		{
			for (int j = 0; j < count[k]; j++)
			{
				int a = coarse_index[k][j]; // Get point ID
				// Position pointer after norm and PQ codes for this level
				char *cur_loc = index_[k] + j * size_per_element_ + sizeof(int) + 2 * sizeof(float) + sizeof(unsigned long) * m_level + i * (M2 + sizeof(float) + sizeof(unsigned long) * m_level) + M2 + sizeof(float);

				// Copy hash codes
				for (int jj = 0; jj < m_level; jj++)
				{
					memcpy(cur_loc, &(code_num[a][jj]), sizeof(unsigned long));
					cur_loc += sizeof(unsigned long);
				}
			}
		}
	}

	// Allocate memory for all centroids in the multilevel - this is a 3d array : [level*subspace][centroid][dimension]
	float ***vec_pq = new float **[size];
	for (int i = 0; i < size; i++)
		vec_pq[i] = new float *[L];
	for (int i = 0; i < size; i++)
		for (int j = 0; j < L; j++)
			vec_pq[i][j] = new float[d2];

	// Read all PQ codebooks from the index file
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < L; j++)
		{
			input_index.read((char *)(vec_pq[i][j]), sizeof(float) * d2);
		}
	}

	input_index.close();

	// Setup some dimension variables
	int half_d = d / 2; // Halves of d for coarse
	float half_m = m / 2; // Half of hash bits (not used)

	// Allocate distance lookup tables for the coarse quantizers
	float *table_1 = new float[L]; // For first half of d
	float *table_2 = new float[L]; // For second half of d

	// Allocate distance lookup tables for the PQ centroids
	// This 3D array stores distances from query to each centroid:
	// [refinement_level][subspace][centroid]
	float ***table2 = new float **[level];
	for (int i = 0; i < level; i++)
	{
		table2[i] = new float *[M2];
	}
	for (int i = 0; i < level; i++)
		for (int j = 0; j < M2; j++)
			table2[i][j] = new float[L];

	// Storage of collision number
	char *col_num = new char[n_pts];

	// For storing query hash codes
	unsigned long *query_proj = new unsigned long[m_level];

	// Constants for distance scaling and numeric conversion ?
	float INV1 = 1.5;
	int INV0 = 200;
	float INV = INV1 / INV0; // Scaling factor ?

	// Counters?
	int nnum = 0;
	int nnum2 = 0;

	// Arrays for cell sorting and processing?
	int *tttest = new int[n_cand1]; // Will hold sortable keys for each cell?
	bool *is_zero = new bool[n_cand1]; // Flags for negative distances?

	// Constants for converting distances to integer keys
	// Large values ensure minimal collision in hashed distances
	int left_inv0 = 100000000; // Scale factor
	int left_inv1 = 100000;	// Modulo value for extracting cell ID


	StopW stopw = StopW();

	// Memory offsets for navigating the compact index structure
	int offset00 = sizeof(float) + m_level * sizeof(unsigned long);
	int offset1 = m_level * sizeof(unsigned long);
	int offset2 = M2 + m_level * sizeof(unsigned long);
	int offset3 = sizeof(float) + M2 + m_level * sizeof(unsigned long);

	int round1_offset = sizeof(float) + sizeof(unsigned long) * m_level + sizeof(float);

	// Main search loop - process each query vector
	for (int i = 0; i < n_query; i++)
	{
		// Get a fresh visited list for tracking processed points
		VisitedList *vl = visited_list_pool_->getFreeVisitedList();
		vl_type *visited_array = vl->mass;
		vl_type visited_array_tag = vl->curV;

		float tau = u[i]; // Distance threshold for this query
		int num = 0; // Counter for results in the final list
		int num1 = 0; // Counter for first-pass candidates
		float *query1 = query[i]; // Current query vector

		// Compute LSH hash codes for the query
		int zero_count = 0;
		for (int j = 0; j < m_level; j++)
		{
			query_proj[j] = 0;
			zero_count = 0;
			for (int jj = 0; jj < m_num; jj++)
			{
				// Project query onto random vectors
				float ttmp0 = compare_ip(query1, proj_array[j * m_num + jj], d);
			
				// Set bit if projection is positive
				if (ttmp0 >= 0)
				{
					query_proj[j] += 1;
					zero_count++;
				}
				// Shift bits for next projection
				if (jj < m_num - 1)
					query_proj[j] = query_proj[j] << 1;
			}
		}

		// Get pointer to second half of query vector
		float *query2 = query1 + half_d;

		// Precompute distances to first-half coarse quantizer centroids
		for (int j = 0; j < L; j++)
		{
			float sum = compare_short(query1, vec1[j], half_d);
			table_1[j] = sum;
		}

		// Precompute distances to second-half coarse quantizer centroids
		for (int j = 0; j < L; j++)
		{
			float sum = compare_short(query2, vec2[j], half_d);
			table_2[j] = sum;
		}

		// Precompute distances to PQ centroids for all levels and subspaces
		for (int j = 0; j < level; j++)
		{
			for (int l = 0; l < M2; l++)
			{
				for (int k = 0; k < L; k++)
				{
					table2[j][l][k] = compare_short(query1 + l * d2, vec_pq[j * M2 + l][k], d2); //! aware <- what's mean by this?
				}
			}
		}
		
		// Calculate distance from query to each coarse quantization cell
		for (int j = 0; j < n_cand1; j++)
		{
			// Get cell's centroid IDs
			unsigned char a = pq_M2[j].id1;
			unsigned char b = pq_M2[j].id2;

			// Calculate combined distance (minus threshold)
			float tmp = table_1[a] + table_2[b] - tau;

			// Convert distance to sortable integer key
			int tmp0;
			if (tmp < 0)
			{
				// Handle negative distances (special encoding)
				tmp0 = (int)(-1 * tmp * left_inv0);
				tmp0 = tmp0 - (tmp0 % left_inv1);
				tmp0 += j;
				is_zero[j] = true;
			}

			else
			{
				// Handle positive distances
				tmp0 = (int)(tmp * left_inv0);
				tmp0 = tmp0 - (tmp0 % left_inv1);
				tmp0 += j;
				is_zero[j] = false;
			}
			tttest[j] = tmp0; // Store sortable key
		}

		// Sort cells by their distance to query
		qsort(tttest, n_cand1, sizeof(int), comp_int);

		// First-pass candidate collection
		int s = 0; // Count points examined
		bool flag = false; // Flag for early termination? ?

		// Calculate offset for accessing PQ codes in memory layout
		int remain_size = size_per_element_ - sizeof(int) - (3 * sizeof(float)) - (m_level * sizeof(unsigned long));

		// Process cells in order of increasing distance
		for (int j = 0; j < n_cand1; j++)
		{
			// Extract cell ID from sortable key
			int a = tttest[j] % left_inv1;

			 // Convert integer key back to float distance
			float cur_dist = 1.0f * (tttest[j] - a) / left_inv0;
			if (is_zero[a] == true) // If dist is negative then flip sign
				cur_dist = -1 * cur_dist;

			int b = count[a]; // Number of points in this cell

			char *cur_obj = index_[a];
			for (int l = 0; l < b; l++)
			{
				s++;
				// Early termination if examined too many points
				if (s > thres_pq)
				{
					flag = true;
					break;
				}

				// Extract point ID and residual norm
				int x = *((int *)cur_obj);
				cur_obj += sizeof(int);
				float y = *((float *)cur_obj);
				char *cur_obj_1 = cur_obj + sizeof(float);
				cur_obj = cur_obj_1 + round1_offset;

				// Calculate approximate distance using PQ
				float z = pq_dist((unsigned char *)cur_obj, table2[0], M2);
				cur_obj += remain_size;

				// combine coarse and fine distances?
				float distance = cur_dist + z * y;

				// Store combined distance for later use
				memcpy(cur_obj_1, &distance, sizeof(float));

				if (distance < 0)
					distance = -1 * distance; // Use absolute value

				// Create neighbor entry and add to candidate set
				Neighbor nn2;
				nn2.id = x;
				nn2.distance = distance;

				if (num1 == 0)
					retset2[0] = nn2;
				else
				{
					if (num1 >= n_exact)
					{
						if (distance >= retset2[n_exact - 1].distance)
							continue;
						InsertIntoPool(retset2.data(), n_exact, nn2);
					}
					else
					{
						InsertIntoPool(retset2.data(), num1, nn2);
					}
				}
				num1++;
			}
			if (flag == true) // Early termination flag means break!
				break;
		}

		// Second-pass: Refine top candidates with exact distance calculation
		for (int j = 0; j < n_exact; j++)
		{
			int ID = retset2[j].id;

			// Skip if already processed
			if (visited_array[ID] == visited_array_tag)
				continue;
			visited_array[ID] = visited_array_tag;

			// Calc exact distance
			float distance = compare_ip(data[ID], query1, d) - tau;
			if (distance < 0)
				distance = -1 * distance;

			// add to final result set
			Neighbor nn2;
			nn2.id = ID;
			nn2.distance = distance;

			if (num == 0)
				retset[0] = nn2;
			else
			{
				if (num >= topk)
				{
					if (distance >= retset[topk - 1].distance)
						continue;
					InsertIntoPool(retset.data(), topk, nn2);
				}
				else
				{
					InsertIntoPool(retset.data(), num, nn2);
				}
			}
			num++;
		}

		// Third-pass: Process remaining clusters with adaptive filtering
		s = 0; // Counter for points examined
		float cur_val = retset[topk - 1].distance; // Current distance threshold for pruning
		bool thres_flag = false; // Flag for seeing when we've examined enough points by soem threshold?

		// Process clusters in order of increasing distance to query
		for (int j = 0; j < n_cand1; j++)
		{

			// Extract cell ID from sortable key
			int a = tttest[j] % left_inv1;
			// Convert integer key back to float distance
			float sort_v = 1.0f * (tttest[j] - a) / left_inv0;
			float v;
			// Handle negative distances - flip sign
			if (is_zero[a] == true)
			{
				v = -1 * sort_v;
			}
			else
			{
				v = sort_v;
			}

			int b = count[a]; // Number of points in this cell

			// Track processing progress for early termination
			if (thres_flag == false)
			{
				if (s + b >= thres_pq)
				{
					thres_flag = true; // Mark that we've examined enough points
				}
				else
				{
					s += b;  // Count these points toward our processed total
				}
			}

			char *cur_obj1 = index_[a]; // Pointer to current cluster's data

			// Examine each point in this cluster
			for (int l = 0; l < b; l++)
			{
				if (thres_flag == true)
					s++; // count individual points when past threshold
				char *cur_obj = cur_obj1; //Curr point pointer
				cur_obj1 += size_per_element_;  // Advance to next point

				// Get point ID and skip if already processed
				int ID = *((int *)cur_obj);
				if (visited_array[ID] == visited_array_tag)
					continue;
				cur_obj += sizeof(int);

				// Get residual norm of this point
				float NORM = *((float *)(cur_obj));
				cur_obj += sizeof(float);

				// Variables for filtering decision
				bool no_exact = false; // Flag to skip exact distance calculation
				float residual_NORM; // Norm of current residual level
				float x = 0; // Distance gap from threshold
				bool is_left = true; // Direction flag (negative vs positive distance)
				float VAL = 0; // Accumulated distance estimate

				// FILTER 1: Quick reject based on coarse distance
				if (sort_v > cur_val) // If cluster distance exceeds curr tresh
				{
					x = sort_v - cur_val; // Calculate gap

					// If gap exceeds threshold, skip this point
					if (x >= delta1 * NORM)
					{
						break; // Skip this is not a canidate
					}

					// If gap is close to threshold, mark for exact distance calc
					else if (s > thres_pq && x >= delta * NORM)
					{

						residual_NORM = NORM;
						if (v >= 0)
							is_left = false;
						cur_obj += offset00;
						goto Label2; // This is a valid candidate - Go to exact distance calc
					}
				}
				
				// FILTER 2: Multi-level refinement with PQ distances
				// This corresponds to the main for loop in Algorithm 2: "for ℓ from 1 to L do"
				int k;
				for (k = 0; k < level; k++)
				{
					// Special handling for first level
					if (k == 0)
					{
						if (s <= thres_pq) // If still in initial sampling phase
						{
							// Use precomputed distance from first pass
							VAL = *(float *)cur_obj;
							cur_obj += offset00;
							residual_NORM = (*(float *)cur_obj) * NORM;

							cur_obj += offset3;
							goto Label; // Skip to distance comparison
						}
						else
						{
							// In adaptive filtering phase
							cur_obj += offset00;
							VAL = v; // Use distance from coarse quantization
						}
					}

					// For each level ℓ, compute accumulated distance approximation
    				// This corresponds to "Compute wℓ(x)" in Algorithm 2
					residual_NORM = (*(float *)cur_obj) * NORM;
					cur_obj += sizeof(float);
					// Add product quantization distance scaled by residual norm
    				// This computes I(x̂ℓ) = I(x̂ℓ-1) + ⟨x̂ℓ,q⟩ from Algorithm 2
					VAL += NORM * pq_dist((unsigned char *)cur_obj, table2[k], M2);
					cur_obj += offset2;

				Label:
					// Compute distance gap 'x' from current threshold
    				// This represents wℓ(x) in Algorithm 2
					if (VAL < 0)
					{
						float ttmp = -1 * VAL;
						x = ttmp - cur_val;
					}
					else
					{
						is_left = false;
						x = VAL - cur_val;
					}

					// If wℓ(x) ≤ 0, point is promising - break to compute exact distance
					if (x <= 0)
					{
						break; // Algorithm 2, line 7-9: "if wℓ(x) ≤ 0 then compute exact distance"
					}
					else
					{
						// If wℓ(x) > rℓ(x), point cannot be in top-k - skip it
        				// Algorithm 2, line 10-11: "if wℓ(x) > rℓ(x) then turn to next data point"
						if (x >= delta1 * residual_NORM)
						{
							no_exact = true;
							break;
						}
						// If Flag=0 and wℓ(x)/rℓ(x) > δ, apply efficiency optimization
        				// Algorithm 2, line 12-13: "if Flag=0 and wℓ(x)/rℓ(x) > δ then skip point"
						else if (x >= delta * residual_NORM)
						{
							break;
						}
						// Otherwise continue to next level
					}
				}

			// This section corresponds to Algorithm 2, lines 14-19: LSH-based filtering
			Label2:
				// When Flag=1 and wℓ(x)/rℓ(x) > δ, or we've reached final level with borderline distance
				if (no_exact == false && x > 0)
				{

					// Compute collision number of data point and query
    				// This is "Compute the collision number of x and q+ or q-" in Algorithm 2
					cur_obj -= offset1;
					int collision_ = 0;

					// Count bit matches between data point and query hash codes
					for (int jj = 0; jj < m_level; jj++)
					{
						collision_ += fast_count(*((unsigned long *)cur_obj), query_proj[jj]);
						cur_obj += sizeof(unsigned long);
					}

					// Calculate angle-based threshold from distance gap
					int y = cosine_inv * x / residual_NORM;
					if (y >= cosine_inv)
						y = cosine_inv - 1;
					y = cosine_table[y];

					// Apply collision testing based on whether point is on left/right side of hyperplane
    				// This is the "if x passes the collision testing" check in Algorithm 2
					if (is_left == true) // For negative side
					{
						if (collision_ >= y) // Too many bit differences
						{
							no_exact = true; // Skip exact calculation
						}
					}
					else // For positive side
					{
						collision_ = m - collision_; // Invert matching count
						if (collision_ >= y)  // Too many bit differences
						{
							no_exact = true; // Skip exact calculation
						}
					}
				}

				// If point passes all filtering tests, compute exact distance
				// This corresponds to "Compute the exact distance of x to H" in Algorithm 2
				if (no_exact == false)
				{
					// Calculate exact inner product distance to hyperplane
					float distance = compare_ip(data[ID], query1, d) - tau;
					if (distance < 0)
						distance = -1 * distance;
					
					// Mark point as visited
					visited_array[ID] = visited_array_tag;
					
					// create neighbor entry
					Neighbor nn2;
					nn2.id = ID;
					nn2.distance = distance;

					// Only add if better than curr kth neighbor
					if (distance >= retset[topk - 1].distance)
						continue;
					else
					{
						// Update top-k neighbors and current distance threshold
        				// This is "Update C and w* if necessary" in Algorithm 2
						InsertIntoPool(retset.data(), topk, nn2);
						cur_val = retset[topk - 1].distance;
					}
				}
			}
			// End of per-cluster processing loop
			// Ensure point counting is correct for early termination
			if (thres_flag == true && s < thres_pq)
			{
				s = thres_pq;
			}
		}

		// Release visited list for reuse
		visited_list_pool_->releaseVisitedList(vl);

		// Copy final results to output array
		for (int j = 0; j < topk; j++)
		{
			search_result[i][j] = retset[j].id;
		}
	}
	// End of search loop - compute overall query time
	float time_us_per_query = stopw.getElapsedTimeMicro() / n_query;

	// Calculate recall by comparing results to ground truth
	int correct = 0;
	for (int i = 0; i < n_query; i++)
	{
		int *massQA2 = massQA + i * maxk;  // Pointer to ground truth for this query
		for (int j = 0; j < real_topk; j++)
		{
			bool real_flag = false;
			for (int l = 0; l < real_topk; l++)
			{
				// Check if result matches ground truth (with offset +1 adjustment)
				if (massQA2[j] == search_result[i][l] + 1)
				{
					correct++; // Count correct matches
					real_flag = true;
					break;
				}
			}
		}
	}
	// Calculate and output final recall and timing metrics
	float recall = 1.0f * correct / n_query / real_topk;
	cout << recall << "\t" << time_us_per_query << " us" << "\n";
}
