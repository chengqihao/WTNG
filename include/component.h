#ifndef STKQ_COMPONENT_H
#define STKQ_COMPONENT_H

#include "index.h"
#include <vector>
#include <limits>
#include <algorithm> 
#include <cmath>


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

        virtual void LoadInner(char *data_emb_file, char *data_loc_file, char *data_video_file, char *query_emb_file, char *query_loc_file, char *query_video_file, char *query_alpha_file, char *ground_file, Parameters &parameters);
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

        void findSkyline(std::vector<Index::WTNGNeighbor> &points, std::vector<Index::WTNGNeighbor> &skyline,
                         std::vector<Index::WTNGNeighbor> &remain_points);

        void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                         std::vector<std::vector<Index::WTNGNeighbor>> &cut_graph_);

        void InsertNode(Index::WTNGNode *qnode, Index::VisitedList *visited_list);

        void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N);

        void SearchAtLayer(Index::WTNGNode *qnode,
                           Index::VisitedList *visited_list,
                           std::vector<Index::NNDescentNeighbor> &result);

        void UpdateEnterpointSet(Index::WTNGNode *qnode);
        void UpdateEnterpointSet();

        void Link(Index::WTNGNode *source, Index::WTNGNode *target, int level, float e_dist, float s_dist, float v_dist);
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
                Index::FurtherFirst f = result.top(); 
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

    // graph conn
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
        // get_neighbors(const float *query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
        get_neighbors(const int query, std::vector<Index::Neighbor> &retset, std::vector<Index::Neighbor> &fullset);
    };

    // select candidate
    class ComponentCandidate : public Component
    {
    public:
        explicit ComponentCandidate(Index *index) : Component(index) {}

        // virtual void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
        //                             std::vector<Index::SimpleNeighbor> &pool) = 0;
    };

    class ComponentCandidateNSG : public ComponentCandidate
    {
    public:
        explicit ComponentCandidateNSG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, unsigned enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::SimpleNeighbor> &result);
    };

    class ComponentCandidateWTNG : public ComponentCandidate
    {
    public:
        explicit ComponentCandidateWTNG(Index *index) : ComponentCandidate(index) {}

        void CandidateInner(unsigned query, std::vector<unsigned> enter, boost::dynamic_bitset<> flags,
                            std::vector<Index::NNDescentNeighbor> &result);
    };

    class ComponentWTNGPruneHeuristic : public ComponentPrune
    {
    public:
        explicit ComponentWTNGPruneHeuristic(Index *index) : ComponentPrune(index) {}

        void findSkyline(std::vector<Index::WTNGNeighbor> &points, std::vector<Index::WTNGNeighbor> &skyline)
        {
            float max_emb_dis = std::numeric_limits<float>::infinity();
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

        void PruneInnerDiscrete(std::vector<Index::NNDescentNeighbor> &pool, unsigned range,
                        std::vector<Index::WTNGNeighbor> &cut_graph_);

        void WTNG2Neighbor(unsigned qnode, unsigned range, std::vector<Index::NNDescentNeighbor> &pool, std::vector<Index::WTNGNeighbor> &result)
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
        // void SearchById_(unsigned query, Index::HnswNode* cur_node, float cur_dist, size_t k,
        //                  size_t ef_search, std::vector<std::pair<Index::HnswNode*, float>> &result);
        void SearchAtLayer(unsigned qnode, Index::HnswNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::FurtherFirst> &result);
    };

    class ComponentSearchRouteWTNG : public ComponentSearchRoute
    {
    public:
        explicit ComponentSearchRouteWTNG(Index *index) : ComponentSearchRoute(index) {}

        void RouteInner(unsigned query, std::vector<Index::Neighbor> &pool, std::vector<unsigned> &res) override;

        void findIndexInAlphaSet2(
            float a1, float a2,
            unsigned int &idx1, unsigned int &idx2,
            const std::vector<float>& alpha_set_1,
            const std::vector<float>& alpha_set_2)
        {
            idx1 = 0;
            idx2 = 0;

            const size_t n = std::min(alpha_set_1.size(), alpha_set_2.size());

            float best1 = std::numeric_limits<float>::infinity();
            float best2 = std::numeric_limits<float>::infinity();

            unsigned int best1_idx = 0;
            unsigned int best2_idx = 1;

            for (size_t k = 0; k < n; ++k)
            {
                float dx = a1 - alpha_set_1[k];
                float dy = a2 - alpha_set_2[k];
                float d2 = dx * dx + dy * dy;  

                if (d2 < best1)
                {
                    best2 = best1; best2_idx = best1_idx;
                    best1 = d2;    best1_idx = static_cast<unsigned int>(k);
                }
                else if (d2 < best2 && k != best1_idx)
                {
                    best2 = d2;
                    best2_idx = static_cast<unsigned int>(k);
                }
            }

            idx1 = best1_idx;
            idx2 = best2_idx;
        }
        

    private:
        void SearchAtLayer(unsigned qnode, Index::WTNGNode *enterpoint, int level,
                           Index::VisitedList *visited_list,
                           std::priority_queue<Index::WTNG_FurtherFirst> &result);
        
        unsigned int alpha_range_index_f;
        unsigned int alpha_range_index_s;
        unsigned int alpha_range_index_t;
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
        // get_neighbors(const float *query, std::vector<Index::Neighbor> &retSet, std::vector<Index::Neighbor> &fullset);
        get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::Neighbor> &retset,
                      std::vector<Index::Neighbor> &fullset);
    };

    class ComponentWTNGRefineEntryCentroid : public ComponentRefineEntry
    {
    public:
        explicit ComponentWTNGRefineEntryCentroid(Index *index) : ComponentRefineEntry(index) {}

        void EntryInner() override;

    private:
        void
        // get_neighbors(const float *query, std::vector<Index::Neighbor> &retSet, std::vector<Index::Neighbor> &fullset);
        get_neighbors(const float *query_emb, const float *query_loc, std::vector<Index::NNDescentNeighbor> &retset,
                      std::vector<Index::NNDescentNeighbor> &fullset);
    };

    class ComponentSearchEntryCentroid : public ComponentSearchEntry
    {
    public:
        explicit ComponentSearchEntryCentroid(Index *index) : ComponentSearchEntry(index) {}

        void SearchEntryInner(unsigned query, std::vector<Index::Neighbor> &pool) override;
    };
}

#endif