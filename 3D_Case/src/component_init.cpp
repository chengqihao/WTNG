#include "component.h"
#include <functional>
#include <atomic>
#include <iomanip>

namespace stkq
{

    void ComponentInitRandom::InitInner()
    {
        SetConfigs(); 

        unsigned range = index->getInitEdgesNum(); 

        index->getFinalGraph().resize(index->getBaseLen()); 

        std::mt19937 rng(rand());
        
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {                                             
            index->getFinalGraph()[i].reserve(range); 
            std::vector<unsigned> tmp(range);         
            GenRandom(rng, tmp.data(), range);        
            for (unsigned j = 0; j < range; j++)
            {
                unsigned id = tmp[j]; 
                if (id == i)
                {
                    continue;
                }

                float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)i * index->getBaseEmbDim(),
                                                         index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                         index->getBaseEmbDim());

                float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)i * index->getBaseLocDim(),
                                                         index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                         index->getBaseLocDim());

                float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                index->getFinalGraph()[i].emplace_back(id, dist);  
            }
            std::sort(index->getFinalGraph()[i].begin(), index->getFinalGraph()[i].end());
        }
    }

    void ComponentInitRandom::SetConfigs()
    {
        index->setInitEdgesNum(index->getParam().get<unsigned>("S"));
    }

    void ComponentInitRandom::GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size)
    {
        unsigned N = index->getBaseLen();

        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size); 
        }

        std::sort(addr, addr + size); 

        for (unsigned i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
    }

    // NSW
    void ComponentInitNSW::InitInner()
    {
        SetConfigs();
        index->nodes_.resize(index->getBaseLen());  // 调用的是HNSW下的public variable
        Index::HnswNode *first = new Index::HnswNode(0, 0, index->NN_, index->NN_);
        index->nodes_[0] = first;
        index->enterpoint_ = first;
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                auto *qnode = new Index::HnswNode(i, 0, index->NN_, index->NN_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
            delete visited_list;
        }
    }

    void ComponentInitNSW::SetConfigs()
    {
        index->NN_ = index->getParam().get<unsigned>("NN");
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
    }

    void ComponentInitNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list)
    {
        Index::HnswNode *enterpoint = index->enterpoint_;

        std::priority_queue<Index::FurtherFirst> result;
        std::priority_queue<Index::CloserFirst> tmp;

        SearchAtLayer(qnode, enterpoint, 0, visited_list, result);

        while (!result.empty())
        {
            tmp.push(Index::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        int pos = 0;
        while (!tmp.empty() && pos < index->NN_)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            Link(top_node, qnode, 0);
            Link(qnode, top_node, 0);
            pos++;
        }
    }

    void ComponentInitNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                         Index::VisitedList *visited_list,
                                         std::priority_queue<Index::FurtherFirst> &result)
    {
        std::priority_queue<Index::CloserFirst> candidates;
        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbData() + (size_t)enterpoint->GetId() * index->getBaseEmbDim(),
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocData() + (size_t)enterpoint->GetId() * index->getBaseLocDim(),
                                                 index->getBaseLocDim());
        float d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

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
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbData() + (size_t)neighbor->GetId() * index->getBaseEmbDim(),
                                                       index->getBaseEmbDim());

                    s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocData() + (size_t)neighbor->GetId() * index->getBaseLocDim(),
                                                       index->getBaseLocDim());
                    d = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentInitNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level)
    {
        source->AddFriends(target, true);
    }

    // HNSW
    void ComponentInitHNSW::InitInner()
    {
        SetConfigs();
        Build(false);
    }

    void ComponentInitHNSW::SetConfigs()
    {
        index->max_m_ = index->getParam().get<unsigned>("max_m");
        index->m_ = index->max_m_;
        index->max_m0_ = index->getParam().get<unsigned>("max_m0");

        auto ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        if (ef_construction_ > 0)
        {
            index->ef_construction_ = ef_construction_;
        }
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
        index->mult = index->getParam().get<int>("mult");
        index->level_mult_ = index->mult > 0 ? index->mult : (1 / log(1.0 * index->m_));
    }

    void ComponentInitHNSW::Build(bool reverse)
    {
        
        index->nodes_.resize(index->getBaseLen());
        int level = GetRandomNodeLevel();
        auto *first = new Index::HnswNode(0, level, index->max_m_, index->max_m0_);  
        index->nodes_[0] = first;
        index->max_level_ = level;
        index->enterpoint_ = first;


        static std::atomic<size_t> g_done{0};
#pragma omp parallel 
        { 
            auto *visited_list = new Index::VisitedList(index->getBaseLen());

            const size_t total = static_cast<size_t>(index->getBaseLen() - 1);

            const size_t step = std::max<size_t>(1000, total / 100);
        
            double last_print = 0.0;
#pragma omp for schedule(dynamic, 128) 
            
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                
                level = GetRandomNodeLevel();
                auto *qnode = new Index::HnswNode(i, level, index->max_m_, index->max_m0_);
                index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list);

                size_t done = g_done.fetch_add(1, std::memory_order_relaxed) + 1;

                if ((done % step == 0) || (done == total)) {
                    #pragma omp critical(progress_print)
                    {
                        double now = omp_get_wtime();
                        static double g_last_wall = 0.0;  
                        if (now - g_last_wall >= 0.5 || done == total) {
                            double pct = 100.0 * double(done) / double(total);
                            std::cout << "\rProgress: " << done << "/" << total
                                    << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                                    << std::flush;
                            g_last_wall = now;
                        }
                    }
                }
            }
            delete visited_list;
        }
        std::cout << std::endl;
    }

    int ComponentInitHNSW::GetRandomNodeLevel()
    {
        static thread_local std::mt19937 rng(GetRandomSeedPerThread());
        static thread_local std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
        double r = uniform_distribution(rng);

        if (r < std::numeric_limits<double>::epsilon())
            r = 1.0;
    
        return (int)(-log(r) * index->level_mult_);
    }

    void ComponentInitHNSW::InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list)
    {
        int cur_level = qnode->GetLevel();
        std::unique_lock<std::mutex> max_level_lock(index->max_level_guard_, std::defer_lock);
        if (cur_level > index->max_level_)
            max_level_lock.lock();

        int max_level_copy = index->max_level_;
        Index::HnswNode *enterpoint = index->enterpoint_;
        if (cur_level < max_level_copy)
        {
            Index::HnswNode *cur_node = enterpoint;

            float e_d, s_d, v_d; 
            if (index->get_alpha_1() != 0)
            {
                e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbData() + (size_t)cur_node->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbDim());
            }
            else
            {
                e_d = 0;
            }

            if (index->get_alpha_2() != 0)
            {

                s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocData() + (size_t)cur_node->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocDim());
            }
            else
            {
                s_d = 0;
            }


            if (index->get_alpha_1() + index->get_alpha_2() != 1)
            {

                v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)qnode->GetId() * index->getBaseVideoDim(),
                                                   index->getBaseVideoData() + (size_t)cur_node->GetId() * index->getBaseVideoDim(),
                                                   index->getBaseVideoDim());
            }
            else
            {
                v_d = 0;
            }

            float d = index->get_alpha_1() * e_d + index->get_alpha_2() * s_d +  (1 - index->get_alpha_1() - index->get_alpha_2()) * v_d;

            float cur_dist = d;
            for (auto i = max_level_copy; i > cur_level; --i)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                    const std::vector<Index::HnswNode *> &neighbors = cur_node->GetFriends(i);

                    for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
                    {
                        if (index->get_alpha_1() != 0)
                        {
                            e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbData() + (size_t)(*iter)->GetId() * index->getBaseEmbDim(),
                                                               index->getBaseEmbDim());
                        }
                        else
                        {
                            e_d = 0;
                        }

                        if (index->get_alpha_2() != 0)
                        {

                            s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocData() + (size_t)(*iter)->GetId() * index->getBaseLocDim(),
                                                               index->getBaseLocDim());
                        }
                        else
                        {
                            s_d = 0;
                        }

                        if (index->get_alpha_1() + index->get_alpha_2() != 1)
                        {

                            v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)qnode->GetId() * index->getBaseVideoDim(),
                                                               index->getBaseVideoData() + (size_t)(*iter)->GetId() * index->getBaseVideoDim(),
                                                               index->getBaseVideoDim());
                        }
                        else
                        {
                            v_d = 0;
                        }

                        d = index->get_alpha_1() * e_d + index->get_alpha_2() * s_d +  (1 - index->get_alpha_1() - index->get_alpha_2()) * v_d;

                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                        }
                    }
                }
            }
            enterpoint = cur_node;
        }

        ComponentPruneHeuristic *a = new ComponentPruneHeuristic(index);
        for (auto i = std::min(max_level_copy, cur_level); i >= 0; --i)
        {
            std::priority_queue<Index::FurtherFirst> result;
            SearchAtLayer(qnode, enterpoint, i, visited_list, result);
            a->Hnsw2Neighbor(qnode->GetId(), index->m_, result); 

            while (!result.empty())
            {
                auto *top_node = result.top().GetNode();
                result.pop();
                Link(top_node, qnode, i);
                Link(qnode, top_node, i);
            }
        }

        if (cur_level > index->enterpoint_->GetLevel())  
        {
            index->enterpoint_ = qnode;
            index->max_level_ = cur_level;
        }
    }

    int ComponentInitHNSW::GetRandomSeedPerThread()
    {
        int tid = omp_get_thread_num();
        int g_seed = 17;
        for (int i = 0; i <= tid; ++i)
            g_seed = 214013 * g_seed + 2531011;
        return (g_seed >> 16) & 0x7FFF;
    }

    
    void ComponentInitHNSW::SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                                          Index::VisitedList *visited_list,
                                          std::priority_queue<Index::FurtherFirst> &result)
    {
        
        std::priority_queue<Index::CloserFirst> candidates;
        float e_d, s_d, v_d;

        if (index->get_alpha_1() != 0)
        {
            e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbData() + (size_t)enterpoint->GetId() * index->getBaseEmbDim(),
                                               index->getBaseEmbDim());
        }
        else
        {
            e_d = 0;
        }

        if (index->get_alpha_2() != 0)
        {

            s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocData() + (size_t)enterpoint->GetId() * index->getBaseLocDim(),
                                               index->getBaseLocDim());
        }
        else
        {
            s_d = 0;
        }

        if (index->get_alpha_1() + index->get_alpha_2() != 1)
        {

            v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)qnode->GetId() * index->getBaseVideoDim(),
                                               index->getBaseVideoData() + (size_t)enterpoint->GetId() * index->getBaseVideoDim(),
                                               index->getBaseVideoDim());
        }
        else
        {
            v_d = 0;
        }


        float d = index->get_alpha_1() * e_d + index->get_alpha_2() * s_d +  (1 - index->get_alpha_1() - index->get_alpha_2()) * v_d;

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

            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    if (index->get_alpha_1() != 0)
                    {
                        e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbData() + (size_t)neighbor->GetId() * index->getBaseEmbDim(),
                                                           index->getBaseEmbDim());
                    }
                    else
                    {
                        e_d = 0;
                    }
                    if (index->get_alpha_2() != 0)
                    {
                        s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocData() + (size_t)neighbor->GetId() * index->getBaseLocDim(),
                                                           index->getBaseLocDim());
                    }
                    else
                    {
                        s_d = 0;
                    }
                    if (index->get_alpha_1() + index->get_alpha_2() != 1)
                    {
                        v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)qnode->GetId() * index->getBaseVideoDim(),
                                                           index->getBaseVideoData() + (size_t)neighbor->GetId() * index->getBaseVideoDim(),
                                                           index->getBaseVideoDim());
                    }
                    else
                    {
                        v_d = 0;
                    }
                    d = index->get_alpha_1() * e_d + index->get_alpha_2() * s_d +  (1 - index->get_alpha_1() - index->get_alpha_2()) * v_d;

                    if (result.size() < index->ef_construction_ || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }


    void ComponentInitHNSW::Link(Index::HnswNode *source, Index::HnswNode *target, int level)
    {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard()); 
        std::vector<Index::HnswNode *> &neighbors = source->GetFriends(level);
        neighbors.push_back(target);
        
        bool shrink = (level > 0 && neighbors.size() > source->GetMaxM()) ||
                      (level <= 0 && neighbors.size() > source->GetMaxM0());
        if (!shrink)
            return;

        std::priority_queue<Index::FurtherFirst> tempres;

        float e_d, s_d, v_d;
        for (const auto &neighbor : neighbors)
        {
            if (index->get_alpha_1() != 0)
            {
                e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)source->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbData() + (size_t)neighbor->GetId() * index->getBaseEmbDim(),
                                                   index->getBaseEmbDim());
            }
            else
            {
                e_d = 0;
            }

            if (index->get_alpha_2() != 0)
            {
                s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)source->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocData() + (size_t)neighbor->GetId() * index->getBaseLocDim(),
                                                   index->getBaseLocDim());
            }
            else
            {
                s_d = 0;
            }

            if (index->get_alpha_1() + index->get_alpha_2() != 1)
            {
                v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)source->GetId() * index->getBaseVideoDim(),
                                                   index->getBaseVideoData() + (size_t)neighbor->GetId() * index->getBaseVideoDim(),
                                                   index->getBaseVideoDim());
            }
            else
            {
                v_d = 0;
            }
            float tmp = index->get_alpha_1() * e_d + index->get_alpha_2() * s_d + (1 - index->get_alpha_1() - index->get_alpha_2()) * v_d;

            tempres.push(Index::FurtherFirst(neighbor, tmp));
        }

        // PRUNE
        ComponentPruneHeuristic *a = new ComponentPruneHeuristic(index);
        a->Hnsw2Neighbor(source->GetId(), tempres.size() - 1, tempres);

        neighbors.clear();  
        while (!tempres.empty())
        {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
        std::priority_queue<Index::FurtherFirst>().swap(tempres);
    }

    void ComponentInitWTNG::InitInner()
    {
        SetConfigs();
        BuildByIncrementInsert();
    }

    void ComponentInitWTNG::SetConfigs()
    {
        index->max_m_ = index->getParam().get<unsigned>("max_m");
        index->m_ = index->max_m_;
        index->ef_construction_ = index->getParam().get<unsigned>("ef_construction");
        index->n_threads_ = index->getParam().get<unsigned>("n_threads");
        index->mult = index->getParam().get<int>("mult");
        index->level_mult_ = index->mult > 0 ? index->mult : (1 / log(1.0 * index->max_m_));
    }

    void ComponentInitWTNG::EntryInner()
    {
        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            index->emb_center[j] = 0; 
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
            {
                index->emb_center[j] += index->getBaseEmbData()[static_cast<size_t>(i) * index->getBaseEmbDim() + j];
            }
        }
        for (unsigned j = 0; j < index->getBaseEmbDim(); j++)
        {
            index->emb_center[j] /= index->getBaseLen();
        }


        for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            index->loc_center[j] = 0;

        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseLocDim(); j++)
            {
                index->loc_center[j] += index->getBaseLocData()[static_cast<size_t>(i) * index->getBaseLocDim() + j];
            }
        }
        for (unsigned j = 0; j < index->getBaseLocDim(); j++)
        {
            index->loc_center[j] /= index->getBaseLen();
        }


        for (unsigned j = 0; j < index->getBaseVideoDim(); j++)
            index->video_center[j] = 0; 

        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getBaseVideoDim(); j++)
            {
                index->video_center[j] += index->getBaseVideoData()[static_cast<size_t>(i) * index->getBaseVideoDim() + j];
            }
        }
        for (unsigned j = 0; j < index->getBaseVideoDim(); j++)
        {
            index->video_center[j] /= index->getBaseLen();
        }
    }

    void ComponentInitWTNG::BuildByIncrementInsert()
    {
        index->WTNG_nodes_.resize(index->getBaseLen());
        int level = 0;
        Index::WTNGNode *first = new Index::WTNGNode(0, index->max_m_);
        index->WTNG_nodes_[0] = first;
        index->WTNG_enterpoints.push_back(first);
        index->emb_center = new float[index->getBaseEmbDim()];  
        index->loc_center = new float[index->getBaseLocDim()];
        index->video_center = new float[index->getBaseVideoDim()];
        
        EntryInner();

        index->max_level_ = level; 

        static std::atomic<size_t> g_done{0};
#pragma omp parallel
        {
            auto *visited_list = new Index::VisitedList(index->getBaseLen()); 

            const size_t total = static_cast<size_t>(index->getBaseLen() - 1);
            const size_t step = std::max<size_t>(1000, total / 100);
            double last_print = 0.0;
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < index->getBaseLen(); ++i)
            {
                level = 0;
                auto *qnode = new Index::WTNGNode(i, index->max_m_);  
                index->WTNG_nodes_[i] = qnode;  
                InsertNode(qnode, visited_list);

                size_t done = g_done.fetch_add(1, std::memory_order_relaxed) + 1;

                if ((done % step == 0) || (done == total)) {
               
                    #pragma omp critical(progress_print)
                    {
                        
                        double now = omp_get_wtime();
                        static double g_last_wall = 0.0;  
                        if (now - g_last_wall >= 0.5 || done == total) {
                            double pct = 100.0 * double(done) / double(total);
                            std::cout << "\rProgress: " << done << "/" << total
                                      << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                                      << std::flush;
                            g_last_wall = now;
                        }
                    }
                }
            }
            delete visited_list;
        }
        std::cout << std::endl;
    }


    void ComponentInitWTNG::UpdateEnterpointSet(Index::WTNGNode *qnode)
    {


        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)qnode->GetId() * index->getBaseEmbDim(),
                                                 index->emb_center,
                                                 index->getBaseEmbDim());

        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)qnode->GetId() * index->getBaseLocDim(),
                                                 index->loc_center,
                                                 index->getBaseLocDim());
        
        float v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)qnode->GetId() * index->getBaseVideoDim(),
                                                 index->video_center,
                                                 index->getBaseVideoDim());

        {
            int index_ep = 0;

            int budget_size = 15;

            std::unique_lock<std::mutex> enterpoint_lock(index->enterpoint_mutex);

            std::vector<Index::NNDescentNeighbor> skyline;

            for (auto it = index->WTNG_enterpoints_skyeline.rbegin(); it != index->WTNG_enterpoints_skyeline.rend(); ++it)
            {
                if ((it->emb_distance_ > e_d) && (it->geo_distance_ > s_d) && (it->video_distance_ > v_d))
                {
                    index_ep = 1;
                    break;
                }
            }

            if (index_ep == 0)
            {
                index->WTNG_enterpoints_skyeline.push_back(Index::NNDescentNeighbor(qnode->GetId(), e_d, s_d, v_d, true, 0));  // vector<NNDescentNeighbor>
                for (auto it = index->WTNG_enterpoints_skyeline.rbegin(); it != index->WTNG_enterpoints_skyeline.rend(); ++it)
                {
                    if ((it->emb_distance_ < e_d) && (it->geo_distance_ < s_d) && (it->video_distance_ < v_d)){
                    }
                    else
                    {
                        skyline.push_back(*it);
                    }
                }

                if (skyline.size() > budget_size)
                {
                    std::size_t k = budget_size / 3;

                    auto topk_by = [&](auto score_getter) {
                        std::vector<Index::NNDescentNeighbor> tmp = skyline;
                        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end(),
                                        [&](const Index::NNDescentNeighbor& a, const Index::NNDescentNeighbor& b) {
                                            return score_getter(a) > score_getter(b); // 取“最大”的 k 个
                                        });
                        tmp.resize(k);
                        return tmp;
                    };

                    auto top_emb = topk_by([](const Index::NNDescentNeighbor& n) { return n.emb_distance_; });
                    auto top_loc = topk_by([](const Index::NNDescentNeighbor& n) { return n.geo_distance_; });
                    auto top_vid = topk_by([](const Index::NNDescentNeighbor& n) { return n.video_distance_; });

                    std::vector<Index::NNDescentNeighbor> out;
                    out.reserve(top_emb.size() + top_loc.size() + top_vid.size());

                    std::unordered_set<int> seen; // key 类型跟 id_ 一致
                    seen.reserve(out.capacity() * 2);

                    for (const auto& x : top_emb) {
                        if (seen.insert(x.id_).second) out.push_back(x);
                    }

                    for (const auto& x : top_loc) {
                        if (seen.insert(x.id_).second) out.push_back(x);
                    }

                    for (const auto& x : top_vid) {
                        if (seen.insert(x.id_).second) out.push_back(x);
                    }
                    out.swap(skyline);
                }

                index->WTNG_enterpoints_skyeline.swap(skyline);

                index->WTNG_enterpoints.clear();

                skyline.clear();

                for (int i = 0; i < index->WTNG_enterpoints_skyeline.size(); i++)
                {
                    index->WTNG_enterpoints.push_back(index->WTNG_nodes_[index->WTNG_enterpoints_skyeline[i].id_]);
                }

            } 
        }
    }

    void ComponentInitWTNG::InsertNode(Index::WTNGNode *qnode, Index::VisitedList *visited_list)
    {

        std::vector<Index::NNDescentNeighbor> pool;

        SearchAtLayer(qnode, visited_list, pool);

        ComponentWTNGPruneHeuristic *a = new ComponentWTNGPruneHeuristic(index);
        std::vector<Index::WTNGNeighbor> result;  
        a->WTNG2Neighbor(qnode->GetId(), qnode->GetMaxM(), pool, result); 
        for (int j = 0; j < result.size(); j++)
        {
            auto *neighbor = index->WTNG_nodes_[result[j].id_];
 
            Link(neighbor, qnode, 0, result[j].emb_distance_, result[j].geo_distance_, result[j].video_distance_);
        }
        qnode->SetFriends(result); 
        
        UpdateEnterpointSet(qnode);

    }


    void ComponentInitWTNG::SearchAtLayer(Index::WTNGNode *qnode,
                                         Index::VisitedList *visited_list,
                                         std::vector<Index::NNDescentNeighbor> &pool)
    {
   
        visited_list->Reset();  
        unsigned ef_construction = index->ef_construction_;
        unsigned query = qnode->GetId();
        
        std::unique_lock<std::mutex> enterpoint_lock(index->enterpoint_mutex, std::defer_lock);
        enterpoint_lock.lock();
        for (int i = 0; i < index->WTNG_enterpoints.size(); i++)  
        {
            auto &enterpoint = index->WTNG_enterpoints[i];
            unsigned enterpoint_id = enterpoint->GetId();

            
            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)query * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + (size_t)enterpoint_id * index->getBaseEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)query * index->getBaseLocDim(),
                                                     index->getBaseLocData() + (size_t)enterpoint_id * index->getBaseLocDim(),
                                                     index->getBaseLocDim());
            
            float v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)query * index->getBaseVideoDim(),
                                                     index->getBaseVideoData() + (size_t)enterpoint_id * index->getBaseVideoDim(),
                                                     index->getBaseVideoDim());
   
            
            pool.emplace_back(enterpoint_id, e_d, s_d, v_d, true, 0);
            
            visited_list->MarkAsVisited(enterpoint_id);
        }

        enterpoint_lock.unlock();
        
        
        if (pool.empty()) {
            unsigned query_id = qnode->GetId();
            float e_d = 0.f, s_d = 0.f, v_d = 0.f;
            pool.emplace_back(query_id, e_d, s_d, v_d, true, 0);
            visited_list->MarkAsVisited(query_id);
            std::cout << "[WTNG][WARN] Enterpoint pool empty. Seeded with query itself: " << query_id << std::endl;
        }

        
        auto queue = Index::skyline_queue(ef_construction);
        queue.init_queue(pool);

        int k = 0;
        int l = 0;
    
        while (k < queue.pool.size())
        {
        
            while (queue.pool[k].layer_ == l)  
            {   
                if (queue.pool[k].flag)
                {
                    queue.pool[k].flag = false;
                    unsigned n = queue.pool[k].id_;
                    Index::WTNGNode *candidate_node = index->WTNG_nodes_[n]; 

                    std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());

                    const std::vector<Index::WTNGNeighbor> &neighbors = candidate_node->GetFriends();  
                    for (unsigned m = 0; m < neighbors.size(); ++m)
                    {
                        unsigned id = neighbors[m].id_;
                        if (visited_list->NotVisited(id))  
                        {
                            visited_list->MarkAsVisited(id);

                            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                                     index->getBaseEmbData() + (size_t)query * index->getBaseEmbDim(),
                                                                     index->getBaseEmbDim());

                            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                                     index->getBaseLocData() + (size_t)query * index->getBaseLocDim(),
                                                                     index->getBaseLocDim());
                            
                            float v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)id * index->getBaseVideoDim(),
                                                                     index->getBaseVideoData() + (size_t)query * index->getBaseVideoDim(),
                                                                     index->getBaseVideoDim());
                            queue.pool.emplace_back(id, e_d, s_d, v_d,  true, -1);
                        }
                    }
                }
                k++;
                if (k >= queue.pool.size())
                {
                    break;
                }
            }
           

            int nk = 0;
            queue.updateNeighbor(nk);   
            k = nk;
            if (k < queue.pool.size())
            {
                l = queue.pool[k].layer_;
            }
        }
        
        pool.swap(queue.pool);
    }

    void ComponentInitWTNG::Link(Index::WTNGNode *source, Index::WTNGNode *target, int level, float e_dist, float s_dist, float v_dist)
    {
        std::unique_lock<std::mutex> lock(source->GetAccessGuard());
        std::vector<Index::WTNGNeighbor> &neighbors = source->GetFriends();
        std::vector<Index::NNDescentNeighbor> tempres;
        std::vector<Index::WTNGNeighbor> result;
        tempres.emplace_back(Index::NNDescentNeighbor(target->GetId(), e_dist, s_dist, v_dist, true, -1));
        for (const auto &neighbor : neighbors)
        {
            tempres.emplace_back(Index::NNDescentNeighbor(neighbor.id_, neighbor.emb_distance_, neighbor.geo_distance_, neighbor.video_distance_, true, -1));
        }
        neighbors.clear();
        ComponentWTNGPruneHeuristic *a = new ComponentWTNGPruneHeuristic(index);
        a->WTNG2Neighbor(source->GetId(), source->GetMaxM(), tempres, result);
        source->SetFriends(result);
        std::vector<Index::NNDescentNeighbor>().swap(tempres);
        std::vector<Index::WTNGNeighbor>().swap(result);
    }
}