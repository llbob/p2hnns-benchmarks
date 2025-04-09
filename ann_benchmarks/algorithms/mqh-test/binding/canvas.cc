# pragma once

#include <vector>


auto point_to_centroid_ID; // point_to_centroid_ID[l][point_id] stores the ID of the centroid to which the point with point_id is assigned at level l.
auto centroid_ips; // centroid_ips[level][centroid_id] stores the distance from the centroid with id centroid_id.
auto is_ip_calculated; // is_ip_calculated maintain a table of booleans, that keeps track of whether the inner product of a given centroid with q is already calculated or not.



std::vector<Neighbor> MQH::query_with_candidates(
    const std::vector<float>& query_pt, 
    int k, 
    float u, 
    int l0, 
    float delta, 
    int query_flag,
    const std::vector<int>& external_candidates)

//precompute inner products of coarse centroids with q
std::vector<float> first_half_ips(L);
std::vector<float> second_half_ips(L);
for(int i = 0; i < L; i++) {
    
}


// Precompute inner products of sub space centroids at remaining levels
std::vector<std::vector<std::vector<float>>> level_ip(
    level, std::vector<std::vector<float>>(M2, std::vector<float>(L)));
    
for (int j = 0; j < level; j++) {
    for (int l = 0; l < M2; l++) {
        for (int k = 0; k < L; k++) {
            level_ip[j][l][k] = compare_short(query.data() + l * (dim / M2), 
            pq_codebooks[j * M2 + l][k].data(), dim / M2);
        }
    }
}

std::vector<Neighbor> candidate_set(k); // result set containing k elements that are updated throughout the pruning process.
float cur_val = 0.0; // current kth nearest neighbour's distance to H

//populate running candidate set and set initial cur_val.
for(int i = 0; i < k; i++)
{
    int point_id = external_candidates.pop_back();
    float distance = compare_short(data[point_id].data(), query.data(), dim) - u;
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



for(int point_id : external_candidates) {
    float ip = 0.0;
    for(int l = 0; l < level; l++){
        // first find the inner product of the centroid and q at this level
        int centroid_id = point_to_centroid_IDs[l][point_id];
        for(int i = 0; i < M2; i++)
        {
            ip += level_ip[l][i][]
        }
        if(is_ip_calculated[l][centroid_id])
        {
            ip = centroid_ips[l][centroid_id];
        }
        else 
        {
            for(int j = 0; j < M2; j++)
            {
                is_ip_calculated[l][centroid_id] = true;
                ip += compare_short(query.data() + j * (dim/M2), pq_codebooks[l*M2+j][centroid_id][j*(dim/M2)], (dim/M2));
                centroid_ips[l][centroid_id] = ip;
            }
        }
        
        // Boolean to check which side of the hyperplane the centroid is situated on



        bool negative_side = ip < -u - cur_val ? true : false;
        if(negative_side) {goto negative}
        else {goto positive}

negative:

        float centroid_dist_to_boundary = 0.0;


        {
            centroid_dist_to_boundary = - u - cur_val - ip;
        } 
        else 
        {
            centroid_dist_to_boundary = - u + cur_val - ip;
        }

        if (centroid_dist_to_boundary < 0) // LINE 7 in pseudocode
        {
            dist_to_H = compare_IP(point, query)
            if(dist_to_H < cur_val)
            {
                InsertIntoPool
                curval = dist_to_H
                break;
            }
        }
        if (centroid_dist_to_boundary > residual_norm) // LINE 10 in pseudocode
        {
            break;
        };

        if (FLAG == 0 && centroid_dist_to_boundary > residual_norm * delta) // LINE 12 in pseudocode
        {
            break;
        }

        if ((FLAG == 1 && centroid_dist_to_boundary > residual_norm * delta) || (FLAG == 0 && l==level-1 centroid_dist_to_boundary <= residual_norm * delta)) // LINE 14 in pseudocode
        {
            // TO DO: Collision testing 
            if(collision)
            {
                dist_to_H = compare_IP(point, query)
                if(dist_to_H < cur_val)
                {
                    InsertIntoPool
                    curval = dist_to_H
                }
            }
        }
    }
}

calculate