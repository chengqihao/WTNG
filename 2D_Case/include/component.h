#ifndef STKQ_COMPONENT_H
#define STKQ_COMPONENT_H

#include "index.h"
#include <array>
#include <utility>
#include <cmath>
#include <limits>
#include <algorithm>
namespace stkq
{
    class Component
    {
    public:
        explicit Component(Index *index) : index(index) {}
        virtual ~Component() { delete index; }

    protected:
        Index *index = nullptr;
    };

    class ComponentLoad : public Component
    {
    public:
        explicit ComponentLoad(Index *index) : Component(index) {}

        virtual void LoadInner(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *query_alpha_file, char *ground_file, Parameters &parameters);
    };

    class ComponentInit : public Component
    {
    public:
        explicit ComponentInit(Index *index) : Component(index) {}

        virtual void InitInner() = 0;
    };

    class ComponentInitRandom : public ComponentInit
    {
    public:
        explicit ComponentInitRandom(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size);
    };

    class ComponentRefine : public Component
    {
    public:
        explicit ComponentRefine(Index *index) : Component(index) {}

        virtual void RefineInner() = 0;
    };

    class ComponentRefineNNDescent : public ComponentRefine
    {
    public:
        explicit ComponentRefineNNDescent(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void SetConfigs();

        void init();

        void NNDescent();

        void join();

        void update();

        void generate_control_set(std::vector<unsigned> &c, std::vector<std::vector<unsigned>> &v, unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned>> &acc_eval_set);
    };

    class ComponentRefineNSG : public ComponentRefine
    {
    public:
        explicit ComponentRefineNSG(Index *index) : ComponentRefine(index) {}

        void RefineInner() override;

    private:
        void Link(Index::SimpleNeighbor *cut_graph_);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, Index::SimpleNeighbor *cut_graph_);

        void SetConfigs();
    };

    class ComponentInitNSW : public ComponentInit
    {
    public:
        explicit ComponentInitNSW(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list);

        void SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list, std::priority_queue<Index::FurtherFirst> &result);

        void Link(Index::HnswNode *source, Index::HnswNode *target, int level);
    };

    class ComponentInitHNSW : public ComponentInit
    {
    public:
        explicit ComponentInitHNSW(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        void Build(bool reverse);

        static int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(Index::HnswNode *qnode, Index::VisitedList *visited_list);

        void SearchAtLayer(Index::HnswNode *qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list, std::priority_queue<Index::FurtherFirst> &result);

        void Link(Index::HnswNode *source, Index::HnswNode *target, int level);
    };

    class ComponentInitDEG : public ComponentInit
    {
    public:
        explicit ComponentInitDEG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        int GetRandomNodeLevel();

        int GetRandomSeedPerThread();

        void BuildByIncrementInsert();

        void init();

        void EntryInner();

        void join();

        void update();

        void Refine();

        void SkylineNNDescent();

        void PruneInner(unsigned n, unsigned range,
                        std::vector<Index::DEGNeighbor> &cut_graph_);

        void findSkyline(std::vector<Index::DEGNeighbor> &points, std::vector<Index::DEGNeighbor> &skyline,
                         std::vector<Index::DEGNeighbor> &remain_points);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                         std::vector<std::vector<Index::DEGNeighbor>> &cut_graph_);

        void InsertNode(Index::DEGNode *qnode, Index::VisitedList *visited_list);

        void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N);

        void SearchAtLayer(Index::DEGNode *qnode,
                           Index::VisitedList *visited_list,
                           std::vector<Index::DEGNNDescentNeighbor> &result);

        void UpdateEnterpointSet(Index::DEGNode *qnode);
        void UpdateEnterpointSet();

        void Link(Index::DEGNode *source, Index::DEGNode *target, int level, float e_dist, float s_dist);

        bool isInRange(float alpha, const std::vector<std::pair<float, float>> &use_range)
        {
            for (const auto &range : use_range)
            {
                if (alpha >= range.first && alpha <= range.second)
                {
                    return true; 
                }
                if (alpha < range.first)
                {
                    return false;
                }
                if (alpha > range.second)
                {
                    continue;
                }
            }
            return false;
        }

        void intersection(const std::vector<std::pair<float, float>> &picked_available_range,
                          const std::pair<float, float> &use_range,
                          std::vector<std::pair<float, float>> &shared_use_range)
        {
            for (const auto &range : picked_available_range)
            {
                if (range.second <= use_range.first)
                {
                    continue;
                }
                else if (range.first >= use_range.second)
                {
                    break;
                }
                else
                {
                    float lower_bound = std::max(range.first, use_range.first);
                    float upper_bound = std::min(range.second, use_range.second);
                    if (lower_bound < upper_bound)
                    {
                        shared_use_range.push_back({lower_bound, upper_bound});
                    }
                }
            }
        }

        void get_use_range(std::vector<std::pair<float, float>> &prune_range, std::vector<std::pair<float, float>> &after_pruned_use_range)
        {
            if (prune_range.empty())
            {
                after_pruned_use_range.push_back(std::make_pair(0.0f, 1.0f));
                return;
            }
            if (prune_range[0].first > 0)
            {
                float gap = prune_range[0].first - 0;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(0, prune_range[0].first));
                }
            }
            int iter;
            for (iter = 0; iter < prune_range.size() - 1; iter++)
            {
                float gap = prune_range[iter + 1].first - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, prune_range[iter + 1].first));
                }
            }
            if (prune_range[iter].second < 1)
            {
                float gap = 1 - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, 1));
                }
            }
            return;
        };

        std::vector<std::pair<float, float>> get_use_range(std::vector<std::pair<float, float>> &intervals)
        {
            std::vector<std::pair<float, float>> use_range;
            if (intervals.empty())
            {
                use_range.push_back(std::make_pair(0, 1));
                return use_range;
            }
            if (intervals[0].first > 0 && intervals[0].first >= 0.01)
            {
                use_range.push_back(std::make_pair(0, intervals[0].first));
            }
            for (size_t i = 0; i < intervals.size() - 1; i++)
            {
                float gap = intervals[i + 1].first - intervals[i].second;
                if (gap >= 0.01)
                {
                    use_range.push_back(std::make_pair(intervals[i].second, intervals[i + 1].first));
                }
            }
            if (intervals[intervals.size() - 1].second < 1 && (1 - intervals[intervals.size() - 1].second) >= 0.01)
            {
                use_range.push_back(std::make_pair(intervals[intervals.size() - 1].second, 1));
            }
            return use_range;
        };

        std::vector<std::pair<float, float>> mergeIntervals(std::vector<std::pair<float, float>> &intervals)
        {
            if (intervals.empty())
                return {};

            std::sort(intervals.begin(), intervals.end());

            std::vector<std::pair<float, float>> merged;

            merged.push_back(intervals[0]);

            for (size_t i = 1; i < intervals.size() - 1; i++)
            {
                if (intervals[i].first <= merged.back().second)
                {
                    merged.back().second = std::max(merged.back().second, intervals[i].second);
                }
                else
                {
                    merged.push_back(intervals[i]);
                }
            }
            return merged;
        }
    };

    class ComponentInitWTNG : public ComponentInit
    {
    public:
        explicit ComponentInitWTNG(Index *index) : ComponentInit(index) {}

        void InitInner() override;

    private:
        void SetConfigs();

        int GetRandomNodeLevel();

        int GetRandomSeedPerThread();

        void BuildByIncrementInsert();

        void init();

        void EntryInner();

        void join();

        void update();

        void Refine();

        void SkylineNNDescent();

        void PruneInner(unsigned n, unsigned range,
                        std::vector<Index::WTNGNeighbor> &cut_graph_);

        void findSkyline(std::vector<Index::WTNGNNDescentNeighbor> &points, std::vector<Index::WTNGNNDescentNeighbor> &skyline,
                         std::vector<Index::WTNGNNDescentNeighbor> &remain_points);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                         std::vector<std::vector<Index::WTNGNeighbor>> &cut_graph_);

        void InsertNode(Index::WTNGNode *qnode, Index::VisitedList *visited_list);

        void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N);

        void SearchAtLayer(Index::WTNGNode *qnode,
                           Index::VisitedList *visited_list,
                           std::vector<Index::WTNGNNDescentNeighbor> &result);

        void UpdateEnterpointSet(Index::WTNGNode *qnode);
        void UpdateEnterpointSet();

        void Link(Index::WTNGNode *source, Index::WTNGNode *target, int level, float e_dist, float s_dist);
    };


    class ComponentPrune : public Component
    {
    public:
        explicit ComponentPrune(Index *index) : Component(index) {}
    };

    class ComponentPruneHeuristic : public ComponentPrune
    {
    public:
        explicit ComponentPruneHeuristic(Index *index) : ComponentPrune(index) {}

        void PruneInner(unsigned q, unsigned range, boost::dynamic_bitset<> flags,
                        std::vector<Index::SimpleNeighbor> &pool, Index::SimpleNeighbor *cut_graph_);

        void Hnsw2Neighbor(unsigned query, unsigned range, std::priority_queue<Index::FurtherFirst> &result)
        {
            int n = result.size();
            std::vector<Index::SimpleNeighbor> pool(n);
            std::unordered_map<int, Index::HnswNode *> tmp;

            for (int i = n - 1; i >= 0; i--)
            {
                Index::FurtherFirst f = result.top(); // 最大堆
                pool[i] = Index::SimpleNeighbor(f.GetNode()->GetId(), f.GetDistance());
                tmp[f.GetNode()->GetId()] = f.GetNode();
                result.pop();
            }

            boost::dynamic_bitset<> flags; 

            auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * range]; 

            PruneInner(query, range, flags, pool, cut_graph_);

            for (unsigned j = 0; j < range; j++)
            {
                if (cut_graph_[range * query + j].distance == -1)
                    break;

                result.push(Index::FurtherFirst(tmp[cut_graph_[range * query + j].id], cut_graph_[range * query + j].distance));
            }

            delete[] cut_graph_;

            std::vector<Index::SimpleNeighbor>().swap(pool);
            std::unordered_map<int, Index::HnswNode *>().swap(tmp);
        }    
    };

    class ComponentConn : public Component
    {
    public:
        explicit ComponentConn(Index *index) : Component(index) {}

        virtual void ConnInner() = 0;
    };

    class ComponentConnNSGDFS : ComponentConn
    {
    public:
        explicit ComponentConnNSGDFS(Index *index) : ComponentConn(index) {}

        void ConnInner();

    private:
        void tree_grow();

        void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

        void findroot(boost::dynamic_bitset<> &flag, unsigned &root);

        void
       
        get_neighbors(const int query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
    };

    class ComponentCandidate : public Component
    {
    public:
        explicit ComponentCandidate(Index *index) : Component(index) {}
    };

    class ComponentCandidateNSG : public ComponentCandidate
    {
    public:
        explicit ComponentCandidateNSG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result);
    };

    class ComponentCandidateDEG : public ComponentCandidate
    {
    public:
        explicit ComponentCandidateDEG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, std::vector<unsigned> enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::DEGNNDescentNeighbor> &result);
    };

    class ComponentDEGPruneHeuristic : public ComponentPrune
    {
    public:
        explicit ComponentDEGPruneHeuristic(Index *index) : ComponentPrune(index) {}

        void intersection(const std::vector<std::pair<float, float>> &picked_available_range,
                          const std::pair<float, float> &use_range,
                          std::vector<std::pair<float, float>> &shared_use_range)
        {
            for (const auto &range : picked_available_range)
            {
                if (range.second <= use_range.first)
                {
                    continue;
                }
                else if (range.first >= use_range.second)
                {
                    break;
                }
                else
                {
                    float lower_bound = std::max(range.first, use_range.first);
                    float upper_bound = std::min(range.second, use_range.second);
                    if (lower_bound < upper_bound)
                    {
                        shared_use_range.push_back({lower_bound, upper_bound});
                    }
                }
            }
        }

        void get_use_range(std::vector<std::pair<float, float>> &prune_range, std::vector<std::pair<float, float>> &after_pruned_use_range)
        {
            if (prune_range.empty())
            {
                after_pruned_use_range.push_back(std::make_pair(0.0f, 1.0f));
                return;
            }
            if (prune_range[0].first > 0)
            {
                float gap = prune_range[0].first - 0;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(0, prune_range[0].first));
                }
            }
            int iter;
            for (iter = 0; iter < prune_range.size() - 1; iter++)
            {
                float gap = prune_range[iter + 1].first - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, prune_range[iter + 1].first));
                }
            }
            if (prune_range[iter].second < 1)
            {
                float gap = 1 - prune_range[iter].second;
                if (gap >= 0.01)
                {
                    after_pruned_use_range.push_back(std::make_pair(prune_range[iter].second, 1));
                }
            }
            return;
        };

        std::vector<std::pair<float, float>> get_use_range(std::vector<std::pair<float, float>> &intervals)
        {
            std::vector<std::pair<float, float>> use_range;
            if (intervals.empty())
            {
                use_range.push_back(std::make_pair(0, 1));
                return use_range;
            }
            if (intervals[0].first > 0 && intervals[0].first >= 0.01)
            {
                use_range.push_back(std::make_pair(0, intervals[0].first));
            }
            for (size_t i = 0; i < intervals.size() - 1; i++)
            {
                float gap = intervals[i + 1].first - intervals[i].second;
                if (gap >= 0.01)
                {
                    use_range.push_back(std::make_pair(intervals[i].second, intervals[i + 1].first));
                }
            }
            if (intervals[intervals.size() - 1].second < 1 && (1 - intervals[intervals.size() - 1].second) >= 0.01)
            {
                use_range.push_back(std::make_pair(intervals[intervals.size() - 1].second, 1));
            }
            return use_range;
        };

        std::vector<std::pair<float, float>> mergeIntervals(std::vector<std::pair<float, float>> &intervals)
        {
            if (intervals.empty())
                return {};

            std::sort(intervals.begin(), intervals.end());

            std::vector<std::pair<float, float>> merged;

            merged.push_back(intervals[0]);

            for (size_t i = 1; i < intervals.size() - 1; i++)
            {
                if (intervals[i].first <= merged.back().second)
                {
      
                    merged.back().second = std::max(merged.back().second, intervals[i].second);
                }
                else
                {
                    merged.push_back(intervals[i]);
                }
            }
            return merged;
        }

        float crossProduct(const Index::DEGNeighbor &O, const Index::DEGNeighbor &A, const Index::DEGNeighbor &B)
        {
            float result = (A.geo_distance_ - O.geo_distance_) * (B.emb_distance_ - O.emb_distance_) - (A.emb_distance_ - O.emb_distance_) * (B.geo_distance_ - O.geo_distance_);
            return result;
        };

        void lowerConvexHull(std::vector<Index::DEGNeighbor> &points, std::vector<Index::DEGNeighbor> &L)
        {
            std::vector<Index::DEGNeighbor> hull;
            for (int i = 0; i < points.size(); i++)
            {
                while (hull.size() >= 2 && (crossProduct(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) <= 0))
                {
                    hull.pop_back();
                }
                hull.push_back(points[i]);
            }
            L.push_back(hull[0]);
            for (int i = 1; i < hull.size(); ++i)
            {
                float deltaY = hull[i].emb_distance_ - hull[i - 1].emb_distance_;
                float deltaX = hull[i].geo_distance_ - hull[i - 1].geo_distance_;
                if (deltaX * deltaY > 0)
                {
                    break;
                }
                else
                {
                    L.push_back(hull[i]);
                }
            }
        };

        void findSkyline(std::vector<Index::DEGNeighbor> &points, std::vector<Index::DEGNeighbor> &skyline)
        {
            float max_emb_dis = std::numeric_limits<float>::max();
            for (const auto &point : points)
            {
                if (point.emb_distance_ < max_emb_dis)
                {
                    skyline.push_back(point);
                    max_emb_dis = point.emb_distance_;
                }
            }
        }

        void PruneInner(std::vector<Index::DEGNNDescentNeighbor> &pool, unsigned range,
                        std::vector<Index::DEGNeighbor> &cut_graph_);

        void DEG2Neighbor(unsigned qnode, unsigned range, std::vector<Index::DEGNNDescentNeighbor> &pool, std::vector<Index::DEGNeighbor> &result)
        {
            PruneInner(pool, range, result);
        };
    };

    class ComponentWTNGPruneHeuristic : public ComponentPrune
    {
    public:
        explicit ComponentWTNGPruneHeuristic(Index *index) : ComponentPrune(index) {}

        void findSkyline(std::vector<Index::WTNGNeighbor> &points, std::vector<Index::WTNGNeighbor> &skyline)
        {
            float max_emb_dis = std::numeric_limits<float>::max();
            for (const auto &point : points)
            {
                if (point.emb_distance_ < max_emb_dis)
                {
                    skyline.push_back(point);
                    max_emb_dis = point.emb_distance_;
                }
            }
            // O(n)
        }

        void PruneInnerDiscrete(std::vector<Index::WTNGNNDescentNeighbor> &pool, unsigned range,
                        std::vector<Index::WTNGNeighbor> &cut_graph_);

        void DEG2Neighbor(unsigned qnode, unsigned range, std::vector<Index::WTNGNNDescentNeighbor> &pool, std::vector<Index::WTNGNeighbor> &result)
        {
            PruneInnerDiscrete(pool, range, result);
        };
    };


    class ComponentSearchRoute : public Component
    {
    public:
        explicit ComponentSearchRoute(Index *index) : Component(index) {}

        virtual void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) = 0;
    };

    class ComponentSearchRouteGreedy : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteGreedy(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;
    };

    class ComponentSearchRouteNSW : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteNSW(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::FurtherFirst> &result);
    };

    class ComponentSearchRouteHNSW : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteHNSW(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

    private:
        void SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::FurtherFirst> &result);
    };

    class ComponentSearchRouteDEG : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteDEG(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

        bool isInRange(float alpha, const std::vector<std::pair<float, float>> &use_range)
        {
            for (const auto &range : use_range)
            {
                if (alpha >= range.first && alpha <= range.second)
                {
                    return true; 
                }
                if (alpha < range.first)
                {
                    return false;
                }
                if (alpha > range.second)
                {
                    continue;
                }
            }
            return false;
        }

    private:
        void SearchAtLayer(unsigned qnode, Index::DEGNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::DEG_FurtherFirst> &result);
    };


    class ComponentSearchRouteWTNG : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteWTNG(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

        void findIndexInAlphaSet(float alpha, unsigned int  &index_d, const std::vector<float>& alpha_set)
        {
            index_d = 0;
            float best_dist = std::numeric_limits<float>::infinity();
            for (size_t i1 = 0; i1 < alpha_set.size(); ++i1)
            {
                float d1 = alpha - alpha_set[i1];
                d1 = d1*d1;
                if (d1 < best_dist) {
                    best_dist = d1;
                    index_d = i1;
                }
            }
        }
        void findIndexInAlphaSet2(float alpha, unsigned int  &index_f, unsigned int &index_s, const std::vector<float>& alpha_set)
        {
            index_f = 6;
            index_s = 6;
            float best_dist = std::numeric_limits<float>::infinity();
            for (size_t i1 = 0; i1 < alpha_set.size(); ++i1)
            {
                float d1 = alpha - alpha_set[i1];
                float d2 = d1*d1;
                if (d2 < best_dist) {
                    best_dist = d2;
                    index_f = i1;
                }
            }
            if (alpha - alpha_set[index_f] < 0 ){
                index_s = index_f - 1;
            }
            else if (alpha - alpha_set[index_f] > 0){
                index_s = index_f + 1;
            }
            else{
                index_s = index_f;
            }
        }
    private:
        void SearchAtLayer(unsigned qnode, Index::WTNGNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::WTNG_FurtherFirst> &result);
        unsigned int alpha_range_index;
        unsigned int alpha_range_index2;
    };

    class ComponentSearchEntry : public Component
    {
    public:
        explicit ComponentSearchEntry(Index *index) : Component(index) {}

        virtual void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) = 0;
    };


    class ComponentSearchEntryNone : public ComponentSearchEntry
    {
    public:
        explicit ComponentSearchEntryNone(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };

    // entry
    class ComponentRefineEntry : public Component
    {
    public:
        explicit ComponentRefineEntry(Index *index) : Component(index) {}

        virtual void EntryInner() = 0;
    };

    class ComponentRefineEntryCentroid : public ComponentRefineEntry
    {
    public:
        explicit ComponentRefineEntryCentroid(Index *index) : ComponentRefineEntry(index) {}

        void EntryInner() override;

    private:
        void
        get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::Neighbor> &retset,
                      std::vector<Index::Neighbor> &fullset);
    };

    class ComponentDEGRefineEntryCentroid : public ComponentRefineEntry
    {
    public:
        explicit ComponentDEGRefineEntryCentroid(Index *index) : ComponentRefineEntry(index) {}

        void EntryInner() override;

    private:
        void
        get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::DEGNNDescentNeighbor> &retset,
                      std::vector<Index::DEGNNDescentNeighbor> &fullset);
    };

    class ComponentSearchEntryCentroid : public ComponentSearchEntry
    {
    public:
        explicit ComponentSearchEntryCentroid(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };
}

#endif