//
// Created by MurphySL on 2020/10/23.
//

#include "component.h"
#include <array>
#include <numeric>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <sstream>


namespace stkq
{
    void ComponentPruneHeuristic::PruneInner(unsigned query, unsigned int range, boost::dynamic_bitset<> flags,
        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_)
        {
            std::vector<Index::SimpleNeighbor> picked;
            for (int i = 0; i < pool.size(); i++)
            {
            bool skip = false;
            float cur_dist = pool[i].distance;
            for (size_t j = 0; j < picked.size(); j++)
            {

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)picked[j].id * index->getBaseEmbDim(),
                                index->getBaseEmbData() + (size_t)pool[i].id * index->getBaseEmbDim(),
                                index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)picked[j].id * index->getBaseLocDim(),
                                index->getBaseLocData() + (size_t)pool[i].id * index->getBaseLocDim(),
                                index->getBaseLocDim());

            float v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)picked[j].id * index->getBaseVideoDim(),
                                index->getBaseVideoData() + (size_t)pool[i].id * index->getBaseVideoDim(),
                                index->getBaseVideoDim());

            float dist = index->get_alpha_1() * e_d + index->get_alpha_2() * s_d + (1 - index->get_alpha_1() - index->get_alpha_2()) * v_d;

            if (dist < cur_dist)
            {
                skip = true;
                break;
            }
            }

            if (!skip)
            {
                picked.push_back(pool[i]);
            }


            if (picked.size() == range)
                break;
            }

            Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range; // 定位到 cut_graph_ 中对应query的邻居起点
            for (size_t t = 0; t < picked.size(); t++)
            {

            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
            }


            if (picked.size() < range)
            {
            des_pool[picked.size()].distance = -1;
            }

            std::vector<Index::SimpleNeighbor>().swap(picked);
        }


        bool checkPoolLayerStepwiseFromZero(const std::vector<Index::NNDescentNeighbor> &pool) {
            if (pool.empty()) return true;
        
            if (pool.front().layer_ != 0) {
                std::cerr << "[WTNG][ERROR] pool.layer_ should start from 0, got "
                          << pool.front().layer_ << std::endl;
                return false;
            }
        
            for (size_t i = 1; i < pool.size(); ++i) {
                int prev = pool[i-1].layer_;
                int curr = pool[i].layer_;
        
                if (curr < prev) {
                    std::cerr << "[WTNG][ERROR] pool.layer_ decreased at index " << i
                              << " (prev=" << prev << ", curr=" << curr << ")" << std::endl;
                    return false;
                }
                if (curr - prev > 1) {
                    std::cerr << "[WTNG][ERROR] pool.layer_ jumped more than +1 at index " << i
                              << " (prev=" << prev << ", curr=" << curr << ")" << std::endl;
                    return false;
                }
            }
        
            return true;
        }

    void ComponentWTNGPruneHeuristic::PruneInnerDiscrete(
        std::vector<Index::NNDescentNeighbor> &pool,
        unsigned int range,
        std::vector<Index::WTNGNeighbor> &cut_graph_)
    {

        std::vector<Index::WTNGNeighbor> picked;  
    
        Index::skyline_queue queue;
    
        queue.init_queue(pool);
   
        pool.swap(queue.pool);
    
       
        if (!checkPoolLayerStepwiseFromZero(pool)) {
            throw std::runtime_error("Pool layer_ sequence is not non-decreasing!");
        }
    

        float RNG_tol = 1;
        int reselect_index = 1;
        int iter = 0;


        while (reselect_index)
        {
            iter = 0;
            std::vector<Index::WTNGNeighbor>().swap(picked);
            
            while (picked.size() < range && iter < (int)pool.size())
            {
    
                int cur_layer = pool[iter].layer_;
                if (cur_layer < 0) {
                    std::ostringstream oss;
                    oss << "[WTNG][ERROR] PruneInnerDiscrete: encountered unassigned layer (layer_ = "
                        << cur_layer << ") at iter=" << iter
                        << ". Ensure queue.init_queue() assigns valid non-negative layers.";
                    throw std::runtime_error(oss.str());
                }
        
                std::vector<Index::NNDescentNeighbor> candidate; 
                while (iter < (int)pool.size() && pool[iter].layer_ == cur_layer) {
                    candidate.emplace_back(pool[iter]);
                    ++iter;
                }
        
                std::vector<float>  alpha_set_1 =
                    index->getParam().get<std::vector<float> >("alpha_set_1");
                std::vector<float>  alpha_set_2 =
                    index->getParam().get<std::vector<float> >("alpha_set_2");
        
           
                for (int i = 0; i < (int)candidate.size(); ++i)
                {
         
                    std::vector<uint8_t>  available_range_discrete = {
                        1,1,1, 1,1,1, 1,1,1, 1
                    };
        
                    float cur_geo_dist   = candidate[i].geo_distance_;   // s_pq
                    float cur_emb_dist   = candidate[i].emb_distance_;   // e_pq
                    float cur_video_dist = candidate[i].video_distance_; // v_pq
        
                    for (size_t j = 0; j < picked.size(); ++j)
                    {
                        float xq_e_dist = index->get_E_Dist()->compare(
                            index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
                            index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                            index->getBaseEmbDim());
                        float xq_s_dist = index->get_S_Dist()->compare(
                            index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
                            index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                            index->getBaseLocDim());
                        float xq_v_dist = index->get_V_Dist()->compare(
                            index->getBaseVideoData() + (size_t)picked[j].id_ * index->getBaseVideoDim(),
                            index->getBaseVideoData() + (size_t)candidate[i].id_ * index->getBaseVideoDim(),
                            index->getBaseVideoDim());
        
                        float exist_e_dist = picked[j].emb_distance_; // e_xp
                        float exist_s_dist = picked[j].geo_distance_; // s_xp
                        float exist_v_dist = picked[j].video_distance_; // v_xp
        
                        for (size_t k = 0; k < alpha_set_1.size(); ++k) {
                            float alpha_1 = alpha_set_1[k];
                            float alpha_2 = alpha_set_2[k];
                        
                            if (picked[j].available_range_discrete[k] == 1) {
                                const float lhs_exist =
                                    alpha_1 * exist_e_dist + alpha_2 * exist_s_dist
                                    + (1 - alpha_1 - alpha_2) * exist_v_dist;
                        
                                const float lhs_xq =
                                    alpha_1 * xq_e_dist + alpha_2 * xq_s_dist
                                    + (1 - alpha_1 - alpha_2) * xq_v_dist;
                        
                                const float rhs_cur =
                                    alpha_1 * cur_emb_dist + alpha_2 * cur_geo_dist
                                    + (1 - alpha_1 - alpha_2) * cur_video_dist;
                        
                                if (lhs_exist <= rhs_cur && RNG_tol * lhs_xq <= rhs_cur) {
                                    available_range_discrete[k] = 0;
                                }
                            }
                        }
                    }
        
                    int sum = std::accumulate(available_range_discrete.begin(),
                                            available_range_discrete.end(),
                                            0);
                    
                    if (sum > 0) {
                        picked.emplace_back(candidate[i].id_,
                                            candidate[i].emb_distance_,
                                            candidate[i].geo_distance_,
                                            candidate[i].video_distance_,
                                            available_range_discrete,
                                            cur_layer);
                        if (picked.size() >= range) break;
                    }
                }
        
                if (picked.size() >= range) break;
            }
            if (picked.size() <= range * 0.8)
            {
                reselect_index += 1;
                if (reselect_index>3)
                {
                    reselect_index = 0;
                }
            }
            else
            {
                reselect_index = 0;  
            }
            RNG_tol = RNG_tol * 1.1;
        }
        cut_graph_.swap(picked);
    }

}
