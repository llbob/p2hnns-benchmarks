# pragma once

#include <vector>
//====================================================================================================================================
//Signature

// std::vector<Neighbor> MQH::query_with_candidates(
//     const std::vector<float>& query_pt, 
//     int k, 
//     float u, 
//     int l0, 
//     float delta, 
//     int query_flag,
//     const std::vector<int>& external_candidates)
//====================================================================================================================================
//
// Preprocessing in query phase. Compute hash code and inner products of centroids with query normal
//
//____________________________________________________________________________________________________________________________________
// Normalize query normal and declare b = - u / query_norm

float query_norm = calc_norm(query.data(), dim);
std::vector<float> normalized_query(dim);
for(int i = 0; i < dim; i++) {
    normalized_query[i] = query.data()[i]/query_norm;
}
float b = -u / query_norm;


//____________________________________________________________________________________________________________________________________
// Precompute inner products of coarse centroids with q

std::vector<float> first_half_ips(L);
std::vector<float> second_half_ips(L);

int half_dim = dim/2;
for (int j = 0; j < L; j++) {
    first_half_ips[j] = compare_short(normalized_query.data(), coarse_centroids_first_half[j].data(), half_dim);
}

// Second half
for (int j = 0; j < L; j++) {
    second_half_ips[j] = compare_short(normalized_query.data() + dim / 2, coarse_centroids_second_half[j].data(), half_dim);
}
//____________________________________________________________________________________________________________________________________
// Precompute inner products of sub space centroids at remaining levels

std::vector<std::vector<std::vector<float>>> level_ip(
    level, std::vector<std::vector<float>>(M2, std::vector<float>(L)));

int sub_dim = dim/M2;
for (int j = 0; j < level; j++) {
    for (int l = 0; l < M2; l++) {
        for (int k = 0; k < L; k++) {
            level_ip[j][l][k] = compare_short(normalized_query.data() + l * sub_dim, 
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
        positive_q_ip += _normalized_query.data()[ll] * proj_array[l][ll];
        negative_q_ip += -normalized_query.data()[ll] * proj_array[l][ll];
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
float cur_val = 0.0; // current kth nearest neighbour's distance to H

//populate running candidate set and set initial cur_val.
for(int i = 0; i < k; i++)
{
    int point_id = external_candidates.pop_back();
    float distance = compare_short(data[point_id].data(), normalized_query.data(), dim) + b;
    if (distance < 0) {
        distance = -distance;
    }

    //Create neighbor instance
    Neighbor nb;
    nb.id = point_id;
    nb.distance = distance;

    //add to candidate set at right location in PQ
    InsertIntoPool(candidate_set.data(), k, nn);

    //update current kth nearest neighbor distance if needed
    if (distance > cur_val)
    {
        cur_val = distance;
    }
}

//====================================================================================================================================
// Begin MQH pruning process starting by the outer for loop in pseudocode

for(int point_id : external_candidates) {
    // skip point id for now
    char *cur_loc = &index_[point_id * size_per_element_] + sizeof(int);
    // initialize current residual norm. Since the normalized residuals of level l-1 are quantized in level l, we need the residual norm of the previous level as scale factor.
    float current_residual_norm = *reinterpret_cast<float>(cur_loc);
    //Skip 2 floats because of VAL
    cur_loc += 2 * sizeof(float);
    // find coarse centroid IDs and initialize IP by looking up the precomputed ip in each sub space.
    unsigned char first_coarse_id = *reinterpret_cast<unsigned char>(cur_loc);
    curloc += sizeof(unsigned char);
    unsigned char second_coarse_id = *reinterpret_cast<unsigned char>(cur_loc);
    curloc += sizeof(unsigned char);
    float ip = first_half_ips[first_coarse_id] + second_half_ips[second_coarse_id];
    
    // gradual refinement of quantization
    for(int l = 0; l < level; l++){
        //find the right memory location for the point at this level by skipping data already read. Therefore, offset = coarse data + previous levels' data
        int offset = sizeof(int) + 2 * (sizeof(float) + sizeof(unsigned char)) + l * (M2 + sizeof(float) + sizeof(unsigned long));
        cur_loc = &index[point_id * size_per_element_] + offset;
        // first update the inner product based on centroid at this level
        for(int i = 0; i < M2; i++)
        {
            // read one codeword at a time and add corresponding precomputed ip to running ip
            unsigned char codeword = *reinterpret_cast<unsigned char>(cur_loc);
            ip += level_ip[l][i][codeword] * current_residual_norm;
            curloc += sizeof(unsigned char);
        }

        if (ip > b - cur_val && ip < b + cur_val) {
            // Centroid lies within boundaries, so x is a promising candidate who's exact distance to H we calculate
            float dist_to_H = compare_short(data[point_id].data(), normalized_query.data(), dim) - u;
            if (dist_to_H < 0) {
                dist_to_H = - dist_to_H;
            }
            if(dist_to_H < cur_val)
            {
                Neighbor nn;
                nn.id = point_id;
                nn.val = point_val;
                InsertIntoPool(candidate_set.data(), k, nn)
                cur_val = candidate_set[k-1];
                break;
            }
        }
        
        // Read residual norm
        float new_residual_norm = *reinterpret_cast<float>(cur_loc);
        cur_loc += sizeof(float);
        
        // Boolean to check which side of the hyperplane the centroid is situated on
        bool positive_side = ip < b - cur_val ? true : false;

        float centroid_dist_to_boundary = 0.0;
        if(positive_side) {
            centroid_dist_to_boundary = b - cur_val - ip;
        }
        else {
            centroid_dist_to_boundary = b + cur_val - ip;
        }


        if (centroid_dist_to_boundary > new_residual_norm) // LINE 10 in pseudocode
        {
            // distance from centroid to bouondary is greater than residual norm, so residual cannot by any means reach inside the margin.
            break;
        }

        if (FLAG == 0 && centroid_dist_to_boundary > new_residual_norm * delta) { // LINE 12 in pseudocode
            // ratio between centroid's distance to boundary and residual_norm is too large, so we prune for efficiency
            break;
        }

        if ((FLAG == 1 && centroid_dist_to_boundary > new_residual_norm * delta) || (FLAG == 0 && l==level-1 centroid_dist_to_boundary <= residual_norm * delta)) // LINE 14 in pseudocode
        {
            // TO DO: Collision testing 

            //First establish bucket with t_zero and t_one and P_zero and P_one
            float t_zero = centroid_dist_to_boundary/residual_norm;
            float t_one = (b + cur_val + ip)/residual_norm;

            float pi_float = static_cast<float>(M_PI);
            float P_zero = 1 - (acos(t_zero)/pi_float);
            float P_one = 1 - (acos(t_one)/pi_float);

            int lower_collision_boundary = P_zero * m_num - l0/2;
            int upper_collision_boundary = P_one * m_num + l0/2;

            //Then read stored bit string for given point at given level
            unsigned long point_bit_string = *reinterpret_cast<unsigned long>(cur_loc);

            // get collision number between query and point
            int collision_number;
            if(positive_side) {
                collision_number = fast_count(point_bit_string, query_bit_string_pos)
            }
            else {
                collision_number = fast_count(point_bit_string, query_bit_string_neg)
            }

            if(collision_number > lower_collision_boundary && collision_number < upper_collision_boundary) {
                float dist_to_H = compare_short(data[point_id].data(), normalized_query.data(), dim) - u;
                if (dist_to_H < 0) {
                    dist_to_H = - dist_to_H;
                }
                if(dist_to_H < cur_val)
                {
                    Neighbor nn;
                    nn.id = point_id;
                    nn.val = point_val;
                    InsertIntoPool(candidate_set.data(), k, nn)
                    cur_val = candidate_set[k-1];
                    break;
                }
            }
        }
        current_residual_norm = new_residual_norm;
    }
}

