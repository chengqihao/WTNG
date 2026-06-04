#include "component.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace stkq
{
    namespace
    {
        constexpr float kVBaseWeightEpsilon = 1e-6f;

        float ModalityDistance(Index *index, unsigned query, unsigned id, unsigned modality)
        {
            if (modality == 0)
            {
                index->addDistCount();
                return index->get_E_Dist()->compare(
                    index->getQueryEmbData() + (size_t)query * index->getBaseEmbDim(),
                    index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                    index->getBaseEmbDim());
            }
            if (modality == 1)
            {
                index->addDistCount();
                return index->get_S_Dist()->compare(
                    index->getQueryLocData() + (size_t)query * index->getBaseLocDim(),
                    index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                    index->getBaseLocDim());
            }

            index->addDistCount();
            return index->get_V_Dist()->compare(
                index->getQueryVideoData() + (size_t)query * index->getBaseVideoDim(),
                index->getBaseVideoData() + (size_t)id * index->getBaseVideoDim(),
                index->getBaseVideoDim());
        }

        float WeightedDistance(Index *index, unsigned query, unsigned id, const std::array<float, 3> &weights)
        {
            float distance = 0.0f;
            for (unsigned modality = 0; modality < weights.size(); ++modality)
            {
                if (weights[modality] > kVBaseWeightEpsilon)
                {
                    distance += weights[modality] * ModalityDistance(index, query, id, modality);
                }
            }
            return distance;
        }

        float KthDistance(const std::unordered_map<unsigned, float> &candidates, unsigned K)
        {
            if (candidates.size() < K)
            {
                return std::numeric_limits<float>::infinity();
            }

            std::vector<float> distances;
            distances.reserve(candidates.size());
            for (const auto &candidate : candidates)
            {
                distances.push_back(candidate.second);
            }
            std::nth_element(distances.begin(), distances.begin() + K - 1, distances.end());
            return distances[K - 1];
        }

        void ExtractTopK(const std::unordered_map<unsigned, float> &candidates, unsigned K, std::vector<unsigned> &res)
        {
            std::vector<std::pair<unsigned, float>> ranked;
            ranked.reserve(candidates.size());
            for (const auto &candidate : candidates)
            {
                ranked.emplace_back(candidate.first, candidate.second);
            }

            std::sort(ranked.begin(), ranked.end(),
                      [](const auto &lhs, const auto &rhs)
                      {
                          if (lhs.second == rhs.second)
                          {
                              return lhs.first < rhs.first;
                          }
                          return lhs.second < rhs.second;
                      });

            res.clear();
            const unsigned result_size = std::min<unsigned>(K, ranked.size());
            res.reserve(result_size);
            for (unsigned i = 0; i < result_size; ++i)
            {
                res.push_back(ranked[i].first);
            }
        }

        void SearchHnswStream(ComponentSearchEntry *entry,
                              ComponentSearchRoute *route,
                              unsigned query,
                              unsigned depth,
                              std::vector<unsigned> &stream)
        {
            std::vector<Index::Neighbor> pool;
            entry->SearchEntryInner(query, pool);
            route->RouteInner(query, pool, stream);
            if (stream.size() > depth)
            {
                stream.resize(depth);
            }
        }

        struct VBaseColumnStream
        {
            std::vector<unsigned> ids;
            unsigned next_pos = 0;
            bool active = false;
        };

        bool HasOpenVBaseStream(const std::array<VBaseColumnStream, 3> &streams)
        {
            for (const auto &stream : streams)
            {
                if (stream.active && stream.next_pos < stream.ids.size())
                {
                    return true;
                }
            }
            return false;
        }
    }

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

        auto *visited_list = new Index::VisitedList(index->getBaseLen()); 

        Index::HnswNode *enterpoint = index->enterpoint_;
        std::vector<std::pair<Index::HnswNode *, float>> ensure_k_path_; 

        Index::HnswNode *cur_node = enterpoint;
        float alpha_1 = index->get_alpha_1();
        float alpha_2 = index->get_alpha_2();
        float e_d, s_d, v_d;
        if (alpha_1 != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)query * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + (size_t)cur_node->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {
            e_d = 0;
        }

        if (alpha_2 != 0)
        {
            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)query * index->getBaseLocDim(),
                                               index->getBaseLocData() + (size_t)cur_node->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }

        if (alpha_1 + alpha_2 != 1)
        {
            v_d = index->get_V_Dist()->compare(index->getQueryVideoData() + (size_t)query * index->getBaseVideoDim(),
                                               index->getBaseVideoData() + (size_t)cur_node->GetId() * index->getBaseVideoDim(),
                                               index->getBaseVideoDim());
        }
        else
        {
            v_d = 0;
        }       
        float d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1 - alpha_2) * v_d;

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

                        if (alpha_1 != 0)
                        {
                            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)query * index->getBaseEmbDim(),
                                                               index->getBaseEmbData() + (size_t)(*iter)->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbDim());
                        }
                        else
                        {
                            e_d = 0;
                        }
                        if (alpha_2 != 0)
                        {

                            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)query * index->getBaseLocDim(),
                                                               index->getBaseLocData() + (size_t)(*iter)->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocDim());
                        }
                        else
                        {
                            s_d = 0;
                        }

                        if (alpha_1 + alpha_2 != 1)
                        {

                            v_d = index->get_V_Dist()->compare(index->getQueryVideoData() + (size_t)query * index->getBaseVideoDim(),
                                                               index->getBaseVideoData() + (size_t)(*iter)->GetId() * index->getBaseVideoDim(),
                                                               index->getBaseVideoDim());
                        }
                        else
                        {
                            v_d = 0;
                        }
                        d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1- alpha_2) * v_d;

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

        float alpha_1 = index->get_alpha_1();
        float alpha_2 = index->get_alpha_2();
        float e_d, s_d, v_d;
        if (alpha_1 != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + (size_t)enterpoint->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {

            e_d = 0;
        }
        if (alpha_2 != 0)
        {

            s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                               index->getBaseLocData() + (size_t)enterpoint->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }
        if (alpha_1 + alpha_2 != 1)
        {

            v_d = index->get_V_Dist()->compare(index->getQueryVideoData() + (size_t)qnode * index->getBaseVideoDim(),
                                               index->getBaseVideoData() + (size_t)enterpoint->GetId() * index->getBaseVideoDim(),
                                               index->getBaseVideoDim());
        }
        else
        {
            v_d = 0;
        }
        float d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1- alpha_2) * v_d;

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
                    if (alpha_1 != 0)
                    {
                        e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                           index->getBaseEmbData() + (size_t)neighbor->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbDim());
                    }
                    else
                    {
                        e_d = 0;
                    }
                    if (alpha_2 != 0)
                    {
                        s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                           index->getBaseLocData() + (size_t)neighbor->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocDim());
                    }
                    else
                    {
                        s_d = 0;
                    }
                    if (alpha_1 + alpha_2 != 1)
                    {
                        v_d = index->get_V_Dist()->compare(index->getQueryVideoData() + (size_t)qnode * index->getBaseVideoDim(),
                                                           index->getBaseVideoData() + (size_t)neighbor->GetId() * index->getBaseVideoDim(),
                                                           index->getBaseVideoDim());
                    }
                    else
                    {
                        v_d = 0;
                    }
                    d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1 - alpha_2) * v_d;

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

    ComponentSearchRouteVBASE::ComponentSearchRouteVBASE(Index *loc_index, Index *emb_index, Index *video_index)
        : ComponentSearchRoute(loc_index), emb_index(emb_index), video_index(video_index)
    {
        loc_entry = new ComponentSearchEntryNone(index);
        emb_entry = new ComponentSearchEntryNone(emb_index);
        video_entry = new ComponentSearchEntryNone(video_index);
        loc_route = new ComponentSearchRouteHNSW(index);
        emb_route = new ComponentSearchRouteHNSW(emb_index);
        video_route = new ComponentSearchRouteHNSW(video_index);
    }

    void ComponentSearchRouteVBASE::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                               std::vector<unsigned int> &res)
    {
        const auto K = index->getParam().get<unsigned>("VBASE_K");
        const auto max_depth = index->getParam().get<unsigned>("L_search");
        const int stride = 3;

        const float alpha_1 = index->getQueryWeightData()[query * stride + 0];
        const float alpha_2 = index->getQueryWeightData()[query * stride + 1];
        std::array<float, 3> weights = {alpha_1, alpha_2, 1.0f - alpha_1 - alpha_2};
        for (float &weight : weights)
        {
            if (std::fabs(weight) <= kVBaseWeightEpsilon)
            {
                weight = 0.0f;
            }
        }

        std::array<VBaseColumnStream, 3> streams;
        if (weights[0] > kVBaseWeightEpsilon)
        {
            SearchHnswStream(emb_entry, emb_route, query, max_depth, streams[0].ids);
            streams[0].active = true;
        }
        if (weights[1] > kVBaseWeightEpsilon)
        {
            SearchHnswStream(loc_entry, loc_route, query, max_depth, streams[1].ids);
            streams[1].active = true;
        }
        if (weights[2] > kVBaseWeightEpsilon)
        {
            SearchHnswStream(video_entry, video_route, query, max_depth, streams[2].ids);
            streams[2].active = true;
        }

        std::unordered_map<unsigned, float> candidates;
        std::unordered_set<unsigned> seen;
        unsigned visited_depth = 0;
        unsigned consecutive_drops = 0;
        constexpr unsigned kDefaultVBaseTermCond = 50;
        last_stop_depth = max_depth;

        while (HasOpenVBaseStream(streams))
        {
            for (unsigned modality = 0; modality < streams.size(); ++modality)
            {
                auto &stream = streams[modality];
                if (!stream.active || stream.next_pos >= stream.ids.size())
                {
                    continue;
                }

                const unsigned candidate_id = stream.ids[stream.next_pos++];
                ++visited_depth;
                if (seen.find(candidate_id) != seen.end())
                {
                    continue;
                }
                seen.insert(candidate_id);

                const float score = WeightedDistance(index, query, candidate_id, weights);
                if (candidates.size() >= K && KthDistance(candidates, K) <= score)
                {
                    ++consecutive_drops;
                }
                else
                {
                    consecutive_drops = 0;
                    candidates.emplace(candidate_id, score);
                }

                if (consecutive_drops >= kDefaultVBaseTermCond)
                {
                    last_stop_depth = visited_depth;
                    ExtractTopK(candidates, K, res);
                    return;
                }
            }
        }

        last_stop_depth = visited_depth;

        ExtractTopK(candidates, K, res);
    }

    void ComponentSearchRouteWTNG::RouteInner(unsigned int query, std::vector<Index::Neighbor> &pool,
                                             std::vector<unsigned int> &res)
    {

        const auto K = index->getParam().get<unsigned>("K_search");  
        auto *visited_list = new Index::VisitedList(index->getBaseLen());
        
        visited_list->Reset();
        unsigned visited_mark = visited_list->GetVisitMark();
        unsigned int *visited = visited_list->GetVisited();

        std::priority_queue<Index::WTNG_FurtherFirst> result;
        std::priority_queue<Index::WTNG_CloserFirst> tmp;
        

        std::vector<float> alpha_set_1 = index->getParam().get<std::vector<float>>("alpha_set_1");
        std::vector<float> alpha_set_2 = index->getParam().get<std::vector<float>>("alpha_set_2");

        float alpha_1 = index->get_alpha_1();
        float alpha_2 = index->get_alpha_2();

        
        findIndexInAlphaSet2(alpha_1, alpha_2, alpha_range_index_f, alpha_range_index_s, alpha_set_1, alpha_set_2);

        SearchAtLayer(query, index->WTNG_enterpoint_, 0, visited_list, result);


        while (!result.empty())
        {
            tmp.push(Index::WTNG_CloserFirst(result.top().GetNode(), result.top().GetEmbDistance(), result.top().GetLocDistance(), result.top().GetVideoDistance(), result.top().GetDistance()));
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

    void ComponentSearchRouteWTNG::SearchAtLayer(unsigned qnode, Index::WTNGNode *enterpoint, int level,
                                                Index::VisitedList *visited_list,
                                                std::priority_queue<Index::WTNG_FurtherFirst> &result)
    {
        const auto L = index->getParam().get<unsigned>("L_search");

        std::priority_queue<Index::WTNG_CloserFirst> candidates;
        float alpha_1 = index->get_alpha_1();
        float alpha_2 = index->get_alpha_2();
        visited_list->Reset();

        bool m_first = false;

        for (int i = 0; i < index->enterpoint_set.size(); i++)
        {
            Index::WTNGNode *cur_node = index->WTNG_nodes_[index->enterpoint_set[i]];

            float cur_e_d = index->get_E_Dist()->compare(index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                                         index->getBaseEmbData() + (size_t)cur_node->GetId() * index->getBaseEmbDim(),
                                                         index->getBaseEmbDim());

            index->addDistCount();

            float cur_s_d = index->get_S_Dist()->compare(index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                                         index->getBaseLocData() + (size_t)cur_node->GetId() * index->getBaseLocDim(),
                                                         index->getBaseLocDim());
            index->addDistCount();

            float cur_v_d = index->get_V_Dist()->compare(index->getQueryVideoData() + (size_t)qnode * index->getBaseVideoDim(),
                                                        index->getBaseVideoData() + (size_t)cur_node->GetId() * index->getBaseVideoDim(),
                                                        index->getBaseVideoDim());
            index->addDistCount();

            float cur_dist = alpha_1 * cur_e_d + alpha_2 * cur_s_d + (1 - alpha_1 - alpha_2) * cur_v_d;

            result.emplace(cur_node, cur_e_d, cur_s_d, cur_v_d, cur_dist);
            candidates.emplace(cur_node, cur_e_d, cur_s_d, cur_v_d, cur_dist);

            visited_list->MarkAsVisited(cur_node->GetId());
        }

        while (!candidates.empty())
        {
            const Index::WTNG_CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            Index::WTNGNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            std::vector<Index::WTNGSimpleNeighbor> &neighbors = candidate_node->GetSearchFriends();
            candidates.pop();
            index->addHopCount();
            for (const auto &neighbor : neighbors)
            {
                int neighbor_id = neighbor.id_;

                const auto& flags = neighbor.active_range_discrete; 

                bool search_flag = false;

                bool search_flag2 = false;

                search_flag = flags[alpha_range_index_f];

                search_flag2 = flags[alpha_range_index_s];

                if (search_flag || search_flag2)
                {
                    if (visited_list->NotVisited(neighbor_id))
                    {
                        visited_list->MarkAsVisited(neighbor_id);

                        if (result.size() >= L)
                        {
                            const float threshold = result.top().GetDistance();
                        
                            const float w1 = alpha_1;
                            const float w2 = alpha_2;
                            const float w3 = 1.0f - alpha_1 - alpha_2;
                        
                            struct Mod { int id; float w; };
                            std::array<Mod,3> mods{{ {0,w1}, {1,w2}, {2,w3} }};
                            std::sort(mods.begin(), mods.end(),
                                      [](const Mod& a, const Mod& b){ return a.w > b.w; });
                        
                            float partial = 0.0f;
                            float e_d = 0.0f, s_d = 0.0f, v_d = 0.0f;
                            bool pruned = false;
                        
                            for (const auto& m : mods)
                            {
                                if (m.w <= 0.0f) continue; // 权重为 0 → 跳过
                        
                                if (m.id == 0) {
                                    e_d = index->get_E_Dist()->compare(
                                        index->getQueryEmbData() + (size_t)qnode * index->getBaseEmbDim(),
                                        index->getBaseEmbData() + (size_t)neighbor_id * index->getBaseEmbDim(),
                                        index->getBaseEmbDim());
                                    index->addDistCount();
                                    partial += m.w * e_d;
                                } else if (m.id == 1) {
                                    s_d = index->get_S_Dist()->compare(
                                        index->getQueryLocData() + (size_t)qnode * index->getBaseLocDim(),
                                        index->getBaseLocData() + (size_t)neighbor_id * index->getBaseLocDim(),
                                        index->getBaseLocDim());
                                    index->addDistCount();
                                    partial += m.w * s_d;
                                } else {
                                    v_d = index->get_V_Dist()->compare(
                                        index->getQueryVideoData() + (size_t)qnode * index->getBaseVideoDim(),
                                        index->getBaseVideoData() + (size_t)neighbor_id * index->getBaseVideoDim(),
                                        index->getBaseVideoDim());
                                    index->addDistCount();
                                    partial += m.w * v_d;
                                }
                        
                                if (partial >= threshold) { 
                                    pruned = true; 
                                    break; 
                                }
                            }
                        
                            if (!pruned)
                            {
                                const float d = alpha_1 * e_d + alpha_2 * s_d + (1.0f - alpha_1 - alpha_2) * v_d;
                                if (d < threshold)
                                {
                                    result.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, v_d, d);
                                    candidates.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, v_d, d);
                                    if (result.size() > L) result.pop();
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

                            float v_d = index->get_E_Dist()->compare(index->getQueryVideoData() + (size_t)qnode * index->getBaseVideoDim(),
                                                                     index->getBaseVideoData() + (size_t)neighbor_id * index->getBaseVideoDim(),
                                                                     index->getBaseVideoDim());
                            float d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1 - alpha_2) * v_d;
                            result.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, v_d, d);
                            candidates.emplace(index->WTNG_nodes_[neighbor_id], e_d, s_d, v_d, d);
                            if (result.size() > L)
                                result.pop();
                        }
                    }
                }
            }
        }
    }
}
