#include "component.h"
#include <array>
#include <numeric>
#include <utility>
#include <limits>
namespace stkq
{
    void ComponentSearchRouteGreedy::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                                std::vector<unsigned int> &res)
    {
        const auto L = index->getParam().get<unsigned>("L_search");
        const auto K = index->getParam().get<unsigned>("K_search");
        std::vector<char> flags(index->getBaseLen(), 0);
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;

            if (pool[k].flag)
            {
                pool[k].flag = false;
                unsigned n = pool[k].id;
                index->addHopCount();
                for (unsigned m = 0; m < index->getLoadGraph()[n].size(); ++m)
                {
                    unsigned id = index->getLoadGraph()[n][m];

                    if (flags[id])
                        continue;
                    flags[id] = 1;

                    float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)query * index->getBaseEmbDim(),
                                                             index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                             index->getBaseEmbDim());

                    float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)query * index->getBaseLocDim(),
                                                             index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                             index->getBaseLocDim());

                    float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    index->addDistCount();

                    if (dist >= pool[L - 1].distance)
                        continue;
                    Index::Neighbor nn(id, dist, true);
                    int r = Index::InsertIntoPool(pool.data(), L, nn);

                    if (r < nk)
                        nk = r;
                }
                // lock to here
            }
            if (nk <= k)
            {
                k = nk;
            }
            else
            {
                ++k;
            }
        }

        res.resize(K);
        for (size_t i = 0; i < K; i++)
        {
            res[i] = pool[i].id;
        }
    }

    void ComponentSearchRouteHNSW::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                              std::vector<unsigned int> &res)
    {

        const auto K = index->getParam().get<unsigned>("K_search"); 

        // const auto L = index->getParam().get<unsigned>("L_search");

        auto *visited_list = new Index::VisitedList(index->getBaseLen()); 

        Index::HnswNode *enterpoint = index->enterpoint_;
        std::vector<std::pair<Index::HnswNode *, float>> ensure_k_path_; 

        Index::HnswNode *cur_node = enterpoint;
        float alpha = index->get_alpha();
        float e_d, s_d;
        if (alpha != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)query * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + (size_t)cur_node->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {
            e_d = 0;
        }

        if (alpha != 1)
        {
            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)query * index->getBaseLocDim(),
                                               index->getBaseLocData() + (size_t)cur_node->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }

        float d = alpha * e_d + (1 - alpha) * s_d;

        index->addDistCount();
        float cur_dist = d;

        ensure_k_path_.clear();
        ensure_k_path_.emplace_back(cur_node, cur_dist);

        for (auto i = index->max_level_; i >= 0; --i)
        {
            visited_list->Reset();
            unsigned visited_mark = visited_list->GetVisitMark();
            unsigned int *visited = visited_list->GetVisited();
            visited[cur_node->GetId()] = visited_mark;

            bool changed = true;
            while (changed)
            {
                changed = false;
                std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                const std::vector<Index::HnswNode *> &neighbors = cur_node->GetFriends(i);

                index->addHopCount();
                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
                {
                    if (visited[(*iter)->GetId()] != visited_mark)
                    {
                        visited[(*iter)->GetId()] = visited_mark;

                        if (alpha != 0)
                        {
                            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)query * index->getBaseEmbDim(),
                                                               index->getBaseEmbData() + (size_t)(*iter)->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbDim());
                        }
                        else
                        {
                            e_d = 0;
                        }
                        if (alpha != 1)
                        {

                            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)query * index->getBaseLocDim(),
                                                               index->getBaseLocData() + (size_t)(*iter)->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocDim());
                        }
                        else
                        {
                            s_d = 0;
                        }
                        d = alpha * e_d + (1 - alpha) * s_d;

                        index->addDistCount();
                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                            ensure_k_path_.emplace_back(cur_node, cur_dist);
                        }
                    }
                }
            }
        }

        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        while (result.size() < K && !ensure_k_path_.empty())
        {
            cur_dist = ensure_k_path_.back().second;
            SearchAtLayer(query, ensure_k_path_.back().first, 0, visited_list, result);
            ensure_k_path_.pop_back();
        }

        while (!result.empty())
        {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
        int pos = 0;
        while (!tmp.empty() && pos < K)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos++;
        }

        delete visited_list;
    }

    void ComponentSearchRouteHNSW::SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                                                 Index::VisitedList *visited_list,
                                                 std::priority_queue<Index::FurtherFirst> &result)
    {
        const auto L = index->getParam().get<unsigned>("L_search");

        std::priority_queue<Index::CloserFirst> candidates;

        float alpha = index->get_alpha();
        float e_d, s_d;
        if (alpha != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + (size_t)enterpoint->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {

            e_d = 0;
        }
        if (alpha != 1)
        {

            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                               index->getBaseLocData() + (size_t)enterpoint->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }
        float d = alpha * e_d + (1 - alpha) * s_d;

        index->addDistCount();
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty())
        {
            const Index::CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<Index::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
            index->addHopCount();
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    if (alpha != 0)
                    {
                        e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                           index->getBaseEmbData() + (size_t)neighbor->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbDim());
                    }
                    else
                    {
                        e_d = 0;
                    }
                    if (alpha != 1)
                    {
                        s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                           index->getBaseLocData() + (size_t)neighbor->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocDim());
                    }
                    else
                    {
                        s_d = 0;
                    }
                    d = alpha * e_d + (1 - alpha) * s_d;

                    index->addDistCount();
                    if (result.size() < L || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > L)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentSearchRouteDEG::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                             std::vector<unsigned int> &res)
    {
        const auto K = index->getParam().get<unsigned>("K_search");
        auto *visited_list = new Index::VisitedList(index->getBaseLen());
        float alpha = index->get_alpha();
        visited_list->Reset();
        unsigned visited_mark = visited_list->GetVisitMark();
        unsigned int *visited = visited_list->GetVisited();

        std::priority_queue<Index::DEG_FurtherFirst> result;
        std::priority_queue<Index::DEG_CloserFirst> tmp;

        // while (result.size() < K && !ensure_k_path_.empty())
        // {
        // cur_dist = ensure_k_path_.back().second;
        SearchAtLayer(query, index->DEG_enterpoint_, 0, visited_list, result);
        // ensure_k_path_.pop_back();
        // }

        while (!result.empty())
        {
            tmp.push(Index::DEG_CloserFirst(result.top().GetNode(), result.top().GetEmbDistance(), result.top().GetLocDistance(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
        int pos = 0;
        while (!tmp.empty() && pos < K)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos++;
        }

        delete visited_list;
    }

    void ComponentSearchRouteWTNG::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                             std::vector<unsigned int> &res)
    {
        const auto K = index->getParam().get<unsigned>("K_search");
        auto *visited_list = new Index::VisitedList(index->getBaseLen());
        float alpha = index->get_alpha();
        visited_list->Reset();
        unsigned visited_mark = visited_list->GetVisitMark();
        unsigned int *visited = visited_list->GetVisited();
        std::priority_queue<Index::WTNG_FurtherFirst> result;
        std::priority_queue<Index::WTNG_CloserFirst> tmp;
        const auto& alpha_set = index->get_alphaset();
        
        findIndexInAlphaSet2(alpha, alpha_range_index, alpha_range_index2, alpha_set);

        SearchAtLayer(query, index->WTNG_enterpoint_, 0, visited_list, result);

        while (!result.empty())
        {
            tmp.push(Index::WTNG_CloserFirst(result.top().GetNode(), result.top().GetEmbDistance(), result.top().GetLocDistance(), result.top().GetDistance()));
            result.pop();
        }        
        
        res.resize(K);
        int pos = 0;

        while (!tmp.empty() && pos < K)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos++;
        }

        delete visited_list;
        
    }



    void ComponentSearchRouteDEG::SearchAtLayer(unsigned qnode, Index::DEGNode *enterpoint, int level,
                                                Index::VisitedList *visited_list,
                                                std::priority_queue<Index::DEG_FurtherFirst> &result)
    {
        const auto L = index->getParam().get<unsigned>("L_search");

        std::priority_queue<Index::DEG_CloserFirst> candidates;
        float alpha = index->get_alpha();
        visited_list->Reset();

        bool m_first = false;

        for (int i = 0; i < index->enterpoint_set.size(); i++)
        {
            Index::DEGNode *cur_node = index->DEG_nodes_[index->enterpoint_set[i]];

            float cur_e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                         index->getBaseEmbData() + (size_t)cur_node->GetId() * index->getBaseEmbDim(),
                                                         index->getBaseEmbDim());

            index->addDistCount();

            float cur_s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                         index->getBaseLocData() + (size_t)cur_node->GetId() * index->getBaseLocDim(),
                                                         index->getBaseLocDim());
            index->addDistCount();

            float cur_dist = alpha * cur_e_d + (1 - alpha) * cur_s_d;

            result.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);
            candidates.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);

            visited_list->MarkAsVisited(cur_node->GetId());
        }

        while (!candidates.empty())
        {
            const Index::DEG_CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::DEGNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            std::vector<Index::DEGSimpleNeighbor> &neighbors = candidate_node->GetSearchFriends();
            candidates.pop();
            index->addHopCount();
            for (const auto &neighbor : neighbors)
            {
                int neighbor_id = neighbor.id_;

                const std::vector<std::pair<int8_t, int8_t>> &use_range = neighbor.active_range;
                bool search_flag = false;
                for (int i = 0; i < use_range.size(); i++)
                {
                    if (alpha * 100 >= use_range[i].first && alpha * 100 <= use_range[i].second)
                    {
                        search_flag = true;
                        break;
                    }
                    if (alpha * 100 < use_range[i].first)
                    {
                        break;
                    }
                    if (alpha * 100 > use_range[i].second)
                    {
                        continue;
                    }
                }



                if (search_flag)
                {
                    if (visited_list->NotVisited(neighbor_id))
                    {
                        visited_list->MarkAsVisited(neighbor_id);

                        if (result.size() >= L)
                        {
                            if (m_first)
                            {
                                float threshold = result.top().GetDistance();

                                float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                                         index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                                                         index->getBaseLocDim());

                                if ((1 - alpha) * s_d >= threshold)
                                {
                                    continue;
                                }

                                float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                                         index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                                                         index->getBaseEmbDim());
                                index->addDistCount();

                                float d = alpha * e_d + (1 - alpha) * s_d;

                                if (threshold > d)
                                {
                                    result.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                                    candidates.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                                    if (result.size() > L)
                                        result.pop();
                                }
                            }
                            else
                            {
                                float threshold = result.top().GetDistance();

                                if (alpha <= 0.5)
                                {
                                    float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                                             index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                                                             index->getBaseLocDim());

                                    if ((1 - alpha) * s_d >= threshold)
                                    {
                                        continue;
                                    }

                                    float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                                             index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                                                             index->getBaseEmbDim());
                                    index->addDistCount();

                                    float d = alpha * e_d + (1 - alpha) * s_d;

                                    if (threshold > d)
                                    {
                                        result.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                                        candidates.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                                        if (result.size() > L)
                                            result.pop();
                                    }
                                }
                                else
                                {
                                    float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                                             index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                                                             index->getBaseEmbDim());

                                    if (alpha * e_d >= threshold)
                                    {
                                        continue;
                                    }

                                    float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                                             index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                                                             index->getBaseLocDim());

                                    index->addDistCount();

                                    float d = alpha * e_d + (1 - alpha) * s_d;

                                    if (threshold > d)
                                    {
                                        result.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                                        candidates.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                                        if (result.size() > L)
                                            result.pop();
                                    }
                                }
                            }
                        }
                        else
                        {
                            float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                                     index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                                                     index->getBaseLocDim());

                            float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                                     index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                                                     index->getBaseEmbDim());
                            float d = alpha * e_d + (1 - alpha) * s_d;
                            result.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                            candidates.emplace(index->DEG_nodes_[neighbor_id], e_d, s_d, d);
                            if (result.size() > L)
                                result.pop();
                        }
                    }
                }
            }
        }
    }


    void ComponentSearchRouteWTNG::SearchAtLayer(unsigned qnode, Index::WTNGNode *enterpoint, int level,
                                                Index::VisitedList *visited_list,
                                                std::priority_queue<Index::WTNG_FurtherFirst> &result)
    {
        const auto L = index->getParam().get<unsigned>("L_search");
        
        std::priority_queue<Index::WTNG_CloserFirst> candidates;

        float alpha = index->get_alpha();
        visited_list->Reset();
        float residual_alpha = 1- alpha;
        for (int i = 0; i < index->Discrete_enterpoint_set.size(); i++){
            Index::WTNGNode *cur_node = index->WTNG_nodes_[index->Discrete_enterpoint_set[i]];
            float cur_e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                        index->getBaseEmbData() + (size_t)cur_node->GetId() * index->getBaseEmbDim(),
                                                        index->getBaseEmbDim());
            index->addDistCount();

            float cur_s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                        index->getBaseLocData() + (size_t)cur_node->GetId() * index->getBaseLocDim(),
                                                        index->getBaseLocDim());
            index->addDistCount();
            
            float cur_dist = alpha * cur_e_d + residual_alpha * cur_s_d;

            result.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);
            candidates.emplace(cur_node, cur_e_d, cur_s_d, cur_dist);

            visited_list->MarkAsVisited(cur_node->GetId());            
        }

        while (!candidates.empty())
        {
            const Index::WTNG_CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;  
            
            Index::WTNGNode *candidate_node = candidate.GetNode();
            std::vector<Index::WTNGSimpleNeighbor> &neighbors = candidate_node->GetSearchFriends();
            candidates.pop();
            index->addHopCount();
            
            for (const auto &neighbor : neighbors)
            {
                int neighbor_id = neighbor.id_;

                const auto& flags = neighbor.active_range_discrete;
                
                bool search_flag = false;
                
                bool search_flag2 = false;

                search_flag = flags[alpha_range_index];
                search_flag2 = flags[alpha_range_index2];
 

                if (search_flag || search_flag2)
                {
                    if (visited_list->NotVisited(neighbor_id))
                    {
                        visited_list->MarkAsVisited(neighbor_id);

                        if (result.size() >= L)
                        {  
                            float threshold = result.top().GetDistance();

                            if (alpha <= 0.5)
                            {
                                float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                                        index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                                                        index->getBaseLocDim());

                                if (residual_alpha * s_d >= threshold)
                                {
                                    continue;
                                }

                                float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                                        index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                                                        index->getBaseEmbDim());
                                index->addDistCount();

                                float d = alpha * e_d + residual_alpha * s_d;

                                if (threshold > d)
                                {
                                    result.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, d);
                                    candidates.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, d);
                                    if (result.size() > L)
                                        result.pop();
                                }
                            }
                            else
                            {
                                float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                                        index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                                                        index->getBaseEmbDim());

                                if (alpha * e_d >= threshold)
                                {
                                    continue;
                                }

                                float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                                        index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                                                        index->getBaseLocDim());

                                index->addDistCount();

                                float d = alpha * e_d + residual_alpha * s_d;

                                if (threshold > d)
                                {
                                    result.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, d);
                                    candidates.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, d);
                                    if (result.size() > L)
                                        result.pop();
                                }
                            }
                        }
                        else
                        {
                            float s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                                    index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                                                    index->getBaseLocDim());

                            float e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                                    index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                                                    index->getBaseEmbDim());
                            float d = alpha * e_d + residual_alpha * s_d;
                            result.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, d);
                            candidates.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, d);
                            if (result.size() > L)
                                result.pop();
                        }
                    }
                }                
            }
        }
    }


   

}