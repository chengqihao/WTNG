//
// Created by MurphySL on 2020/10/23.
//

#include "component.h"
#include <array>
#include <numeric>
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

                float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

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

        Index::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range; 
        for (size_t t = 0; t < picked.size(); t++)
        {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
            // std::cout << picked[t].id << "|" << picked[t].distance << " ";
        }

        if (picked.size() < range)
        {
            des_pool[picked.size()].distance = -1;
        }

        std::vector<Index::SimpleNeighbor>().swap(picked);
    }

    void ComponentDEGPruneHeuristic::PruneInner(std::vector<Index::DEGNNDescentNeighbor> &pool, unsigned int range,
                                                     std::vector<Index::DEGNeighbor> &cut_graph_)
    {
        std::vector<Index::DEGNeighbor> picked;
        Index::skyline_queue queue;
        sort(pool.begin(), pool.end());
        queue.init_queue(pool);
        pool.swap(queue.pool);
        int iter = 0;
        int visited_layer = 0;
        while (picked.size() < range && iter < pool.size())
        {
            std::vector<Index::DEGNNDescentNeighbor> candidate;
            while (iter < pool.size())
            {
                if (pool[iter].layer_ == visited_layer)
                {
                    candidate.emplace_back(pool[iter]);
                }
                else
                {
                    break;
                }
                iter++;
            }
            std::vector<Index::DEGNeighbor> tempres_picked;
            for (int i = 0; i < candidate.size(); i++)
            {
                std::vector<std::pair<float, float>> prune_range;
                float cur_geo_dist = candidate[i].geo_distance_; // s_pq
                float cur_emb_dist = candidate[i].emb_distance_; // e_pq
                for (size_t j = 0; j < picked.size(); j++)
                {
                    const std::vector<std::pair<float, float>> &picked_use_range = picked[j].available_range;
                    float xq_e_dist = index->get_E_Dist()->compare(
                        index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                        index->getBaseEmbDim());
                    // E(x,q)

                    float xq_s_dist = index->get_S_Dist()->compare(
                        index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
                        index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                        index->getBaseLocDim());
                    // S(x,q)

                    float exist_e_dist = picked[j].emb_distance_; // e_xp

                    float exist_s_dist = picked[j].geo_distance_; // s_xp
                    float diff1 = exist_e_dist - cur_emb_dist + cur_geo_dist - exist_s_dist;
                    float diff2 = cur_geo_dist - exist_s_dist;

                    std::pair<float, float> tmp_prune_range_1;
                    if (diff1 > 0 && diff2 > 0)
                    {
                        // equation 1 holds on when alpha < = diff2 / diff1
                        tmp_prune_range_1 = std::make_pair(0.0f, std::min(diff2 / diff1, 1.0f));
                        // eq1_prune_lower_alpha = diff2 / diff1 ;
                    }
                    else if (diff1 < 0 && diff2 < 0)
                    {
                        // equation 1 holds on when alpha > = diff2 / diff1
                        // eq1_prune_upper_alpha = diff2 / diff1 ;
                        tmp_prune_range_1 = std::make_pair(std::min(diff2 / diff1, 1.0f), 1.0f);
                    }
                    else if (diff1 < 0 && diff2 > 0)
                    {
                        tmp_prune_range_1 = {0.0f, 1.0f};
                        // the equation hold forever
                    }
                    else if (diff1 > 0 && diff2 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_1 = {0.0f, 0.0f};
                    }
                    // now for equation 2
                    float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;
                    float diff4 = cur_geo_dist - xq_s_dist;
                    /*
                    similar to previous
                    */
                    // when alpha >= eq1_prune_upper_alpha and alpha <= eq1_prune_lower_alpha, the equation holds on
                    // float eq2_prune_upper_alpha = 1;
                    // float eq2_prune_lower_alpha = 0;
                    std::pair<float, float> tmp_prune_range_2;
                    if (diff3 > 0 && diff4 > 0)
                    {
                        // equation 2 holds on when alpha < = diff4 / diff3
                        // eq2_prune_upper_alpha = diff4 / diff3;
                        // tmp_prune_range.second = std::min(tmp_prune_range.second, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(0.0f, std::min(1.0f, diff4 / diff3));
                    }
                    else if (diff3 < 0 && diff4 < 0)
                    {
                        // equation 2 holds on when alpha > = diff4 / diff3
                        // eq2_prune_lower_alpha = diff4 / diff3;
                        // tmp_prune_range.first = std::max(tmp_prune_range.first, diff4 / diff3);
                        tmp_prune_range_2 = std::make_pair(std::min(diff4 / diff3, 1.0f), 1.0f);
                    }
                    else if (diff3 < 0 && diff4 > 0)
                    {
                        // the equation hold forever
                        // then we do not change the previous range
                        tmp_prune_range_2 = {0.0f, 1.0f};
                    }
                    else if (diff3 > 0 && diff4 < 0)
                    {
                        // equation never hold
                        // break;
                        tmp_prune_range_2 = {0.0f, 0.0f};
                    }

                    std::pair<float, float> tmp_prune_range;
                    tmp_prune_range.first = std::max(tmp_prune_range_1.first, tmp_prune_range_2.first);
                    tmp_prune_range.second = std::min(tmp_prune_range_1.second, tmp_prune_range_2.second);

                    if (tmp_prune_range.second > tmp_prune_range.first)
                    {
                        // now we consider whether this range is useful range, that is (second > first)
                        // now we check its intersection range with shared_use_range
                        intersection(picked_use_range, tmp_prune_range, prune_range);
                    }
                    else
                    {
                        continue;
                        // this range is not useful, so this edge will not be pruned by this selected edge
                    }
                }
                prune_range = mergeIntervals(prune_range);
                std::vector<std::pair<float, float>> after_pruned_use_range;
                get_use_range(prune_range, after_pruned_use_range);
                float threshold = 0.1;
                float use_size = 0;
                for (int j = 0; j < after_pruned_use_range.size(); j++)
                {
                    use_size = use_size + after_pruned_use_range[j].second - after_pruned_use_range[j].first;
                }
                if (use_size >= threshold)
                {
                    picked.push_back(Index::DEGNeighbor(candidate[i].id_, candidate[i].emb_distance_,
                                                             candidate[i].geo_distance_, after_pruned_use_range, visited_layer));
                }
            }
            visited_layer++;
            if (picked.size() >= range)
                break;
        }
        cut_graph_.swap(picked);
    }

    void ComponentWTNGPruneHeuristic::PruneInnerDiscrete(std::vector<Index::WTNGNNDescentNeighbor> &pool, unsigned int range,
        std::vector<Index::WTNGNeighbor> &cut_graph_)
    {
        
        Index::skyline_queue_my queue;
        sort(pool.begin(), pool.end());
        queue.init_queue(pool);  
        pool.swap(queue.pool); 

        
        const auto& alpha_set = index->get_alphaset();

        std::vector<Index::WTNGNeighbor> picked; 
        int iter = 0;
        int visited_layer = 0;
        int reselect_index = 1;
        float RNG_tol = 1;
        float alpha_tol = 1;
        while (reselect_index)
        {
            iter = 0;
            visited_layer = 0;
            std::vector<Index::WTNGNeighbor>().swap(picked);
            while (picked.size() < range && iter < pool.size())
            {
                std::vector<Index::WTNGNNDescentNeighbor> candidate; 
                while (iter < pool.size())  
                {
                    if (pool[iter].layer_ == visited_layer)
                    {
                        candidate.emplace_back(pool[iter]);
                    }
                    else 
                    {
                        break;
                    }
                    iter++;
                }

                
                for (int i = 0; i < candidate.size(); i++) 
                {
                    float cur_geo_dist = candidate[i].geo_distance_; 
                    float cur_emb_dist = candidate[i].emb_distance_; 
                    std::vector<uint8_t> available_range_discrete(alpha_set.size(), uint8_t{1});  
                    std::vector<int> count_rng_number(4);
                    

                    for (size_t j = 0; j < picked.size(); j++)  
                    {
                        float xq_e_dist = index->get_E_Dist()->compare(
                            index->getBaseEmbData() + (size_t)picked[j].id_ * index->getBaseEmbDim(),
                            index->getBaseEmbData() + (size_t)candidate[i].id_ * index->getBaseEmbDim(),
                            index->getBaseEmbDim());  
                        float xq_s_dist = index->get_S_Dist()->compare(
                            index->getBaseLocData() + (size_t)picked[j].id_ * index->getBaseLocDim(),
                            index->getBaseLocData() + (size_t)candidate[i].id_ * index->getBaseLocDim(),
                            index->getBaseLocDim()); 
                        float exist_e_dist = picked[j].emb_distance_; 
                        float exist_s_dist = picked[j].geo_distance_; 

                        unsigned index_range = 0;
                        for (size_t l = 0; l < alpha_set.size(); l++)
                        {
                            float alpha = alpha_set[l];
                            if (picked[j].available_range_discrete[l] == 1)
                            {
                                if ((alpha * exist_e_dist + (1-alpha) * exist_s_dist <= alpha * cur_emb_dist + (1-alpha) * cur_geo_dist) && 
                                   RNG_tol * ((alpha * xq_e_dist + (1-alpha) * xq_s_dist) <=  alpha * cur_emb_dist + (1-alpha) * cur_geo_dist))
                                {
                                    available_range_discrete[l] = 0; 
                                }
                            }

                        }
                    }
                    
                    int sum = std::accumulate(available_range_discrete.begin(),available_range_discrete.end(), 0);
                    if (sum > 0)
                    {
                        picked.emplace_back(
                            candidate[i].id_, candidate[i].emb_distance_, candidate[i].geo_distance_,
                            std::move(available_range_discrete), visited_layer);
                    }
  
                }
                visited_layer++;  
                if (picked.size() >= range)
                    break;
            }
            if (picked.size() <= range * 0.7)
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
            alpha_tol = alpha_tol + 0.05;
        }
        cut_graph_.swap(picked);        
    }
}