#ifndef STKQ_INDEX_H
#define STKQ_INDEX_H

#include <omp.h>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <set>
#include <functional>
#include <map>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include "util.h"
#include "distance.h"
#include "parameters.h"
#include "policy.h"
#include "CommonDataStructure.h"
#include <mm_malloc.h>
#include <cstddef>
#include <unordered_map>
#include <limits>
#include <stdlib.h>
#define INF_N -std::numeric_limits<float>::max()  
#define INF_P std::numeric_limits<float>::max()  
namespace stkq
{
    typedef std::lock_guard<std::mutex> LockGuard;
    class NNDescent
    {
    public:
        unsigned K;
        unsigned S;
        unsigned R;
        unsigned L;
        unsigned ITER;
        struct Neighbor
        {
            unsigned id;  
            float distance;  
            bool flag;  

            Neighbor() = default;

            Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}  // {}初始化是风格，与()初始化等价

            inline bool operator<(const Neighbor &other) const
            {
                return distance < other.distance;
            }

            inline bool operator>(const Neighbor &other) const
            {
                return distance > other.distance;
            }
        };

        struct nhood
        {
            std::mutex lock;            
            std::vector<Neighbor> pool; 
            unsigned M;                 

            std::vector<unsigned> nn_old;
            std::vector<unsigned> nn_new;  
            std::vector<unsigned> rnn_old;
            std::vector<unsigned> rnn_new;

            nhood() {}

            nhood(unsigned l, unsigned s)
            {
                M = s;
                nn_new.resize(s * 2);
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N)
            {
                M = s;
                nn_new.resize(s * 2);
                GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N); 
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(const nhood &other)
            {
                M = other.M;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
            }

            void insert(unsigned id, float dist) 
            {
                LockGuard guard(lock);
                if (dist > pool.front().distance)
                    return;
                for (unsigned i = 0; i < pool.size(); i++)
                {
                    if (id == pool[i].id)
                        return;
                }
                if (pool.size() < pool.capacity())
                {
                    pool.push_back(Neighbor(id, dist, true));
                    std::push_heap(pool.begin(), pool.end());
                }
                else
                {
                    std::pop_heap(pool.begin(), pool.end());
                    pool[pool.size() - 1] = Neighbor(id, dist, true);
                    std::push_heap(pool.begin(), pool.end());
                }
            }

            template <typename C>
            void join(C callback) const
            {
                for (unsigned const i : nn_new)
                {
                    for (unsigned const j : nn_new)
                    {
                        if (i < j)
                        {
                            callback(i, j);
                        }
                    }
                    for (unsigned j : nn_old)
                    {
                        callback(i, j);
                    }
                }
            }
        };

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

        typedef std::vector<nhood> KNNGraph;
        KNNGraph graph_;
    };

    class NSW
    {
    public:
        unsigned NN_;  
        unsigned ef_construction_ = 150; 
        unsigned n_threads_ = 32;  
    };

    class HNSW
    {
    public:
        unsigned m_ = 12; 
        unsigned max_m_ = 12;  
        unsigned max_m0_ = 24;  
        int mult;  
        float level_mult_ = 1 / log(1.0 * m_);  

        int max_level_ = 0; 
        mutable std::mutex max_level_guard_;

        template <typename KeyType, typename DataType>
        class MinHeap
        { 
        public:
            class Item
            {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType &key) : key(key) {}
                Item(const KeyType &key, const DataType &data) : key(key), data(data) {}
                bool operator<(const Item &i2) const
                {
                    return key > i2.key;
                }
            };

            MinHeap()
            {
            }

            const KeyType top_key()
            {
                if (v_.size() <= 0)
                    return 0.0;
                return v_[0].key;
            }

            Item top()
            {
                if (v_.size() <= 0)
                    throw std::runtime_error("[Error] Called top() operation with empty heap"); 
                return v_[0];
            }

            void pop()
            {
                std::pop_heap(v_.begin(), v_.end());  
                v_.pop_back(); 
            }

            void push(const KeyType &key, const DataType &data)
            {
                v_.emplace_back(Item(key, data));  
                std::push_heap(v_.begin(), v_.end());  
            }


            size_t size()
            {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };

        class HnswNode
        {
        public:
            explicit HnswNode(int id, int level, size_t max_m, size_t max_m0)
                : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level + 1)
            {
                for (int i = 1; i <= level; ++i)
                    friends_at_layer_[i].reserve(max_m_ + 1);

                friends_at_layer_[0].reserve(max_m0_ + 1);
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) { id_ = id; }
            inline int GetLevel() const { return level_; }
            inline void SetLevel(int level) { level_ = level; }
            inline size_t GetMaxM() const { return max_m_; }
            inline size_t GetMaxM0() const { return max_m0_; }

            inline std::vector<HnswNode *> &GetFriends(int level) { return friends_at_layer_[level]; } // 传引用
            inline void SetFriends(int level, std::vector<HnswNode *> &new_friends)
            {
                if (level >= friends_at_layer_.size())
                    friends_at_layer_.resize(level + 1);
                friends_at_layer_[level].swap(new_friends);  
            }

            inline std::mutex &GetAccessGuard() { return access_guard_; }

            inline void AddFriends(HnswNode *element, bool bCheckForDup)
            {
                std::unique_lock<std::mutex> lock(access_guard_);  
                if (bCheckForDup)
                {
                    auto it = std::lower_bound(friends_at_layer_[0].begin(), friends_at_layer_[0].end(), element);
                    if (it == friends_at_layer_[0].end() || (*it) != element)  
                    {
                        friends_at_layer_[0].insert(it, element);
                    }
                }
                else
                {
                    friends_at_layer_[0].push_back(element); 
                }
            }

        private:
            int id_;
            int level_;
            size_t max_m_;
            size_t max_m0_;

            std::vector<std::vector<HnswNode *>> friends_at_layer_;  
            std::mutex access_guard_;  
        };

        class FurtherFirst
        {
        public:
            FurtherFirst(HnswNode *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode *GetNode() const { return node_; }
            bool operator<(const FurtherFirst &n) const
            {
                return (distance_ < n.GetDistance());
            }

        private:
            HnswNode *node_;
            float distance_;
        };

        class CloserFirst
        {
        public:
            CloserFirst(HnswNode *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode *GetNode() const { return node_; }
            bool operator<(const CloserFirst &n) const
            {
                return (distance_ > n.GetDistance());
            }

        private:
            HnswNode *node_;
            float distance_;
        };

        typedef typename std::pair<HnswNode *, float> IdDistancePair;
        struct IdDistancePairMinHeapComparer
        {
            bool operator()(const IdDistancePair &p1, const IdDistancePair &p2) const
            {
                return p1.second > p2.second;
            }
        };
        typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer>> IdDistancePairMinHeap;
        HnswNode *enterpoint_ = nullptr;
        std::vector<HnswNode *> nodes_;
    };

    class NSG
    {
    public:
        unsigned R_refine;
        unsigned L_refine;
        unsigned C_refine;

        unsigned ep_;
        unsigned width;
    };

    class SSG
    {
    public:
        float A;
        unsigned n_try;

        std::vector<unsigned> eps_;
        unsigned test_min = INT_MAX;
        unsigned test_max = 0;
        long long test_sum = 0;
    };

    

    class WTNG
    {
    public:
        struct WTNGNeighbor 
        {
            unsigned id_;
            float emb_distance_;  
            float geo_distance_;  
            float video_distance_;  
            unsigned layer_;
            std::vector<uint8_t> available_range_discrete; 
            

            WTNGNeighbor() = default;
            WTNGNeighbor(unsigned id, float emb_distance, float geo_distance, float video_distance, std::vector<uint8_t> range_discrete) : id_{id}, 
                emb_distance_{emb_distance}, geo_distance_(geo_distance), video_distance_(video_distance), available_range_discrete(range_discrete) {}
            WTNGNeighbor(unsigned id, float emb_distance, float geo_distance, float video_distance, unsigned l) : id_{id}, 
                emb_distance_{emb_distance}, geo_distance_(geo_distance), video_distance_(video_distance), layer_(l) {}
            WTNGNeighbor(unsigned id, float emb_distance, float geo_distance, float video_distance, std::vector<uint8_t> range_discrete, unsigned l) : id_{id}, 
                emb_distance_{emb_distance}, geo_distance_(geo_distance), video_distance_(video_distance), available_range_discrete(range_discrete), layer_(l) {}

            inline bool operator<(const WTNGNeighbor &other) const
            {
                return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_) || 
                        (geo_distance_ == other.geo_distance_ && emb_distance_ == other.emb_distance_ && video_distance_ < other.video_distance_));
            }
        };

        struct WTNGSimpleNeighbor
        {
            unsigned id_;
            std::vector<uint8_t>  active_range_discrete; 
            WTNGSimpleNeighbor() = default;
            WTNGSimpleNeighbor(unsigned id, std::vector<uint8_t>  range_discrete) : id_{id}, active_range_discrete(range_discrete) {}
        };

        struct NNDescentNeighbor
        {
            unsigned id_;
            float emb_distance_;
            float geo_distance_;
            float video_distance_;  
            bool flag;  
            int layer_;
            NNDescentNeighbor() = default;
            NNDescentNeighbor(unsigned id, float emb_distance, float geo_distance, float video_distance, bool f, int layer) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), video_distance_(video_distance), flag(f), layer_(layer)
            {
            }
            inline bool operator<(const NNDescentNeighbor &other) const
            {
                return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_) || 
                        (geo_distance_ == other.geo_distance_ && emb_distance_ == other.emb_distance_ && video_distance_ < other.video_distance_));
            }
        };


        struct CompareDisEmb
        {
            bool operator()(const NNDescentNeighbor& a, const NNDescentNeighbor& b) const {
                return a.emb_distance_ < b.emb_distance_;
            }
        };

        struct CompareDisGeo
        {
            bool operator()(const NNDescentNeighbor& a, const NNDescentNeighbor& b) const {
                return a.geo_distance_ < b.geo_distance_;
            }
        };

        struct CompareDisVideo
        {
            bool operator()(const NNDescentNeighbor& a, const NNDescentNeighbor& b) const {
                return a.video_distance_ < b.video_distance_;
            }
        };

        class WTNGNode
        {
        public:
            explicit WTNGNode(int id, int max_m)
                : id_(id), max_m_(max_m)
            {
                friends.clear();
                friends_for_search.clear();
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) { id_ = id; }
            inline int GetMaxM() const { return max_m_; }

            inline void SetMaxM(int max_m) { max_m_ = max_m; }
            inline std::vector<WTNGNeighbor> &GetFriends() { return friends; }

            inline void SetFriends(std::vector<WTNGNeighbor> &new_friends)
            {
                friends.swap(new_friends);
            }

            inline std::vector<WTNGSimpleNeighbor> &GetSearchFriends() { return friends_for_search; }

            inline void SetSearchFriends(std::vector<WTNGSimpleNeighbor> &new_friends)
            {
                friends_for_search.swap(new_friends);
            }

            inline std::mutex &GetAccessGuard() { return access_guard_; }

        private:
            int id_;
            size_t max_m_;
            std::vector<WTNGNeighbor> friends;
            std::vector<WTNGSimpleNeighbor> friends_for_search;
            std::mutex access_guard_;
        };

        struct skyline_queue
        {
            std::vector<NNDescentNeighbor> pool;

            unsigned M; 
            unsigned num_layer;
            
            std::vector<float> f2_list;
            std::vector<float> f3_list;

            skyline_queue() {}

            skyline_queue(unsigned l)
            {
                M = l;
                pool.reserve(M);
            }

            skyline_queue(const skyline_queue &other)
            {
                M = other.M;
                pool.reserve(other.pool.capacity());
            }

            void init_queue(std::vector<NNDescentNeighbor> &insert_points)
            {
                std::vector<NNDescentNeighbor> skyline_result;
                std::vector<NNDescentNeighbor> remain_points;

                auto points_eg = insert_points;
                auto points_ev = insert_points;
                auto points_gv = insert_points;
                int l = 0;
                std::sort(points_eg.begin(), points_eg.end(), CompareDisEmb{});
                std::sort(points_ev.begin(), points_ev.end(), CompareDisEmb{});
                std::sort(points_gv.begin(), points_gv.end(), CompareDisGeo{});
                std::unordered_map<unsigned, bool> visited;

                while (!points_eg.empty() || !points_ev.empty() || !points_gv.empty())
                {
        
                    std::size_t added_this_round = 0;
                    if (!points_eg.empty())
                    {
                        findSkyline(points_eg, skyline_result, remain_points, 0);
                        if (skyline_result.empty())
                        {
                            std::cerr << "[Fatal Error] skyline_result is empty at indicator <indicator> (layer=" << l << ")" << std::endl;
                            std::exit(EXIT_FAILURE);
                        }
                        points_eg.swap(remain_points);
                        for (auto &point : skyline_result)
                        {
                            if (!visited[point.id_])
                            {
                                pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.video_distance_, true, l);
                                visited[point.id_] = true;
                                ++added_this_round;
                            }
    
                        }
                        std::vector<NNDescentNeighbor>().swap(skyline_result);
                        std::vector<NNDescentNeighbor>().swap(remain_points);
                    }

                    if (!points_ev.empty())
                    {
                        findSkyline(points_ev, skyline_result, remain_points, 1);
                        if (skyline_result.empty())
                        {
                            std::cerr << "[Fatal Error] skyline_result is empty at indicator <indicator> (layer=" << l << ")" << std::endl;
                            std::exit(EXIT_FAILURE);
                        }
                        points_ev.swap(remain_points);
                        for (auto &point : skyline_result)
                        {
                            if (!visited[point.id_])
                            {
                                pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.video_distance_, true, l);
                                visited[point.id_] = true;
                                ++added_this_round;
                            }
                        }
                        std::vector<NNDescentNeighbor>().swap(skyline_result);
                        std::vector<NNDescentNeighbor>().swap(remain_points);
                    }

                    if (!points_gv.empty())
                    {
                        findSkyline(points_gv, skyline_result, remain_points, 2);
                        if (skyline_result.empty())
                        {
                            std::cerr << "[Fatal Error] skyline_result is empty at indicator <indicator> (layer=" << l << ")" << std::endl;
                            std::exit(EXIT_FAILURE);
                        }
                        points_gv.swap(remain_points);
                        for (auto &point : skyline_result)
                        {
                            if (!visited[point.id_])
                            {
                                pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.video_distance_, true, l);
                                visited[point.id_] = true;
                                ++added_this_round;
                            }
                        }
                        std::vector<NNDescentNeighbor>().swap(skyline_result);
                        std::vector<NNDescentNeighbor>().swap(remain_points);
                    }
                    if (added_this_round == 0) {
                        continue;
                    } else {
                        ++l; 
                    }
                }
                num_layer = l;
                insert_points.clear();
            }


            void findSkyline(std::vector<NNDescentNeighbor> &points, std::vector<NNDescentNeighbor> &skyline, std::vector<NNDescentNeighbor> &remain_points, int indicator)
            {
                if (indicator == 0) 
                {
                    float min_geo_dis = std::numeric_limits<float>::infinity();
                    for (const auto &point : points)
                    {   
                        if (point.geo_distance_ < min_geo_dis)
                        {
                            skyline.push_back(point);
                            min_geo_dis = point.geo_distance_;
                        }
                        else
                        {
                            remain_points.emplace_back(point);
                        }
                    }
                    if (skyline.empty() && !points.empty()) {
                        for (const auto& point : points) {
                            if (point.geo_distance_ == min_geo_dis || !std::isfinite(point.geo_distance_)) {
                                std::cerr << "[Warn] All geo_distance_ invalid (inf or NaN), picking first as fallback\n";
                                std::cout << point.geo_distance_ << std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
                else if (indicator == 1)  // ev
                {
                    float min_video_dis = std::numeric_limits<float>::infinity();
                    for (const auto &point : points)
                    {   
                        if (point.video_distance_ < min_video_dis)
                        {
                            skyline.push_back(point);
                            min_video_dis = point.video_distance_;
                        }
                        else
                        {
                            remain_points.emplace_back(point);
                        }
                    }
                    if (skyline.empty() && !points.empty()) {
                        for (const auto& point : points) {
                            if (point.video_distance_ == min_video_dis || !std::isfinite(point.video_distance_)) {
                                std::cerr << "[Warn] All video_distance_ invalid (inf or NaN), picking first as fallback\n";
                                std::cout << point.video_distance_ << point.geo_distance_ << std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
                else if (indicator == 2) // gv
                {
                    float min_video_dis = std::numeric_limits<float>::infinity();
                    for (const auto &point : points)
                    {   
                        if (point.video_distance_ < min_video_dis)
                        {
                            skyline.push_back(point);
                            min_video_dis = point.video_distance_;
                        }
                        else
                        {
                            remain_points.emplace_back(point);
                        }
                    }
                    if (skyline.empty() && !points.empty()) {
                        
                        for (const auto& point : points) {
                            if (point.video_distance_ == min_video_dis || !std::isfinite(point.video_distance_)) {
                                std::cerr << "[Warn] All video_distance_ invalid (inf or NaN), picking first as fallback\n";
                                
                                std::cout << point.video_distance_ << std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
                // O(3n)
            }

            void updateNeighbor(int &nk)
            {
                std::vector<NNDescentNeighbor> skyline_result;
                std::vector<NNDescentNeighbor> remain_points;
                std::vector<NNDescentNeighbor> candidate_eg;

                candidate_eg.swap(pool); 
                auto candidate_ev = candidate_eg;
                auto candidate_gv = candidate_eg;
                int l = 0;
                int k = 0;
                
                std::sort(candidate_eg.begin(), candidate_eg.end(), CompareDisEmb{});
                std::sort(candidate_ev.begin(), candidate_ev.end(), CompareDisEmb{});
                std::sort(candidate_gv.begin(), candidate_gv.end(), CompareDisGeo{});
                std::unordered_map<unsigned, bool> visited;

                bool updated = true;
                while (pool.size() < M && (candidate_eg.size() > 0 || candidate_ev.size() > 0 || candidate_gv.size() > 0))
                {
                    if (!candidate_eg.empty())
                    {
                        findSkyline(candidate_eg, skyline_result, remain_points, 0);
                        candidate_eg.swap(remain_points);
                        for (auto &point : skyline_result)
                        {
                            if (!visited[point.id_])
                            {
                                pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.video_distance_, point.flag, l);
                                visited[point.id_] = true;
                                if (updated)
                                {
                                    if (point.flag == true)
                                    {
                                        nk = k;
                                        updated = false;
                                    }
                                    else
                                    {
                                        nk++;
                                    }
                                }
                                k++;
                            }    
                        }
                        std::vector<NNDescentNeighbor>().swap(skyline_result);
                        std::vector<NNDescentNeighbor>().swap(remain_points);
                    }

                    if (!candidate_ev.empty())
                    {
                        findSkyline(candidate_ev, skyline_result, remain_points, 1);
                        candidate_ev.swap(remain_points);
                        for (auto &point : skyline_result)
                        {
                            if (!visited[point.id_])
                            {
                                pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.video_distance_, point.flag, l);
                                visited[point.id_] = true;
                                if (updated)
                                {
                                    if (point.flag == true)
                                    {
                                        nk = k;
                                        updated = false;
                                    }
                                    else
                                    {
                                        nk++;
                                    }
                                }
                                k++;
                            }
                        }
                        std::vector<NNDescentNeighbor>().swap(skyline_result);
                        std::vector<NNDescentNeighbor>().swap(remain_points);
                    }

                    if (!candidate_gv.empty())
                    {
                        findSkyline(candidate_gv, skyline_result, remain_points, 2);
                        candidate_gv.swap(remain_points);
                        for (auto &point : skyline_result)
                        {
                            if (!visited[point.id_])
                            {
                                pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.video_distance_, point.flag, l);
                                visited[point.id_] = true;
                                if (updated)
                                {
                                    if (point.flag == true)
                                    {
                                        nk = k;
                                        updated = false;
                                    }
                                    else
                                    {
                                        nk++;
                                    }
                                }
                                k++;
                            }
                        }
                        std::vector<NNDescentNeighbor>().swap(skyline_result);
                        std::vector<NNDescentNeighbor>().swap(remain_points);
                    }

                    l++;
                }
                num_layer = l;
            }
        };

        template <typename KeyType, typename DataType>
        class MaxHeap
        {
        public:
            class Item
            {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType &key) : key(key) {}
                Item(const KeyType &key, const DataType &data) : key(key), data(data) {}
                bool operator<(const Item &i2) const
                {
                    return key < i2.key; 
                }
            };

            MaxHeap()
            {
                std::make_heap(v_.begin(), v_.end());
            }

            const KeyType top_key()
            {
                if (v_.empty())                                                                     
                    throw std::runtime_error("[Error] Called top_key() operation with empty heap"); 
                return v_[0].key;
            }

            Item top()
            {
                if (v_.empty())                                                               
                    throw std::runtime_error("[Error] Called top() operation with empty heap"); 
                return v_[0];
            }
            

            void pop()
            {
                if (v_.empty())                                                                
                    throw std::runtime_error("[Error] Called pop() operation with empty heap"); 
                std::pop_heap(v_.begin(), v_.end());                                           
                v_.pop_back();                                                               
            }
         

            void push(const KeyType &key, const DataType &data)
            {
                v_.emplace_back(Item(key, data));    
                std::push_heap(v_.begin(), v_.end()); 
            }
       

            size_t size()
            {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };


        class WTNG_FurtherFirst
        {
        public:
            WTNG_FurtherFirst(WTNGNode *node, float emb_distance, float geo_distance, float video_distance,float dist) : node_(node), emb_distance_(emb_distance),
            geo_distance_(geo_distance), video_distance_(video_distance), dist_(dist) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetVideoDistance() const { return video_distance_; }
            inline float GetDistance() const { return dist_; }
            inline WTNGNode *GetNode() const { return node_; }
            bool operator<(const WTNG_FurtherFirst &n) const
            {
                return (dist_ < n.GetDistance());
            }

        private:
            WTNGNode *node_;
            float emb_distance_;
            float geo_distance_;
            float video_distance_;
            float dist_;
        };

        class WTNG_CloserFirst
        {
        public:
            WTNG_CloserFirst(WTNGNode *node, float emb_distance, float geo_distance, float video_distance, float dist) : node_(node), emb_distance_(emb_distance), 
            geo_distance_(geo_distance), video_distance_(video_distance), dist_(dist) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetVideoDistance() const { return video_distance_; }
            inline float GetDistance() const { return dist_; }
            inline WTNGNode *GetNode() const { return node_; }
            bool operator<(const WTNG_CloserFirst &n) const
            {
                return (dist_ > n.GetDistance());
            }

        private:
            WTNGNode *node_;
            float emb_distance_;
            float geo_distance_;
            float video_distance_;
            float dist_;
        };

        class WTNG_GeoFurtherFirst
        {
        public:
            WTNG_GeoFurtherFirst(WTNGNode *node, float emb_distance, float geo_distance, float video_distance, float dist) : node_(node), emb_distance_(emb_distance), 
            geo_distance_(geo_distance), video_distance_(video_distance) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetVideoDistance() const { return video_distance_; }
            inline WTNGNode *GetNode() const { return node_; }
            bool operator<(const WTNG_GeoFurtherFirst &n) const
            {
                return (geo_distance_ < n.GetLocDistance() || (geo_distance_ == n.GetLocDistance() && emb_distance_ < n.GetEmbDistance()));
            }

        private:
            WTNGNode *node_;
            float emb_distance_;
            float geo_distance_;
            float video_distance_;
        };

        class WTNG_GeoCloserFirst
        {
        public:
            WTNG_GeoCloserFirst(WTNGNode *node, float emb_distance, float geo_distance, float video_distance) : node_(node), emb_distance_(emb_distance),
            geo_distance_(geo_distance), video_distance_(video_distance){}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetVideoDistance() const { return video_distance_; }
            inline WTNGNode *GetNode() const { return node_; }
            bool operator<(const WTNG_GeoCloserFirst &n) const
            {
                return (geo_distance_ > n.GetLocDistance() || (geo_distance_ == n.GetLocDistance() && emb_distance_ > n.GetEmbDistance()));
            }

        private:
            WTNGNode *node_;
            float emb_distance_;
            float geo_distance_;
            float video_distance_;
        };

        WTNGNode *WTNG_enterpoint_ = nullptr;
        std::vector<WTNGNode *> WTNG_nodes_;
        std::vector<WTNGNode *> WTNG_enterpoints;
        std::vector<NNDescentNeighbor> WTNG_enterpoints_skyeline;

        std::mutex enterpoint_mutex;
        std::vector<unsigned> enterpoint_set;
        unsigned rnn_size;
        float *emb_center, *loc_center, *video_center;
    };

    class Index : public NNDescent, public NSW, public HNSW, public SSG, public NSG, public WTNG
    {
    public:
        explicit Index(float max_emb_dist, float max_spatial_dist, float max_video_dist)
        {
            e_dist_ = new E_Distance(max_emb_dist); // 指针成员
            s_dist_ = new E_Distance(max_spatial_dist);  // 指针成员
            v_dist_ = new E_Distance(max_video_dist);
        }

        ~Index()
        {
            delete e_dist_;
            delete s_dist_;
            delete v_dist_;
        }

        struct SimpleNeighbor
        {
            unsigned id;
            float distance;  

            SimpleNeighbor() = default;
            SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance} {}

            inline bool operator<(const SimpleNeighbor &other) const
            {
                return distance < other.distance;
            }
        };

        typedef std::vector<std::vector<SimpleNeighbor>> FinalGraph;  
        typedef std::vector<std::vector<unsigned>> LoadGraph;

        class VisitedList
        {
        public:
            VisitedList(unsigned size) : size_(size), mark_(1)
            {
                visited_ = new unsigned int[size_];
                memset(visited_, 0, sizeof(unsigned int) * size_);
            }

            ~VisitedList() { delete[] visited_; }

            inline bool Visited(unsigned int index) const { return visited_[index] == mark_; }

            inline bool NotVisited(unsigned int index) const { return visited_[index] != mark_; }

            inline void MarkAsVisited(unsigned int index) { visited_[index] = mark_; }

            inline void Reset()
            {
                if (++mark_ == 0)
                {
                    mark_ = 1;
                    memset(visited_, 0, sizeof(unsigned int) * size_);
                }
            }

            inline unsigned int *GetVisited() { return visited_; }

            inline unsigned int GetVisitMark() { return mark_; }

        private:
            unsigned int *visited_;
            unsigned int size_;
            unsigned int mark_;
        };

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

        float *getBaseEmbData() const
        {
            return base_emb_data_;
        }

        void setBaseEmbData(float *baseEmbData)
        {
            base_emb_data_ = baseEmbData;
        }

        float *getBaseLocData() const
        {
            return base_loc_data_;
        }

        void setBaseLocData(float *baseLocData)
        {
            base_loc_data_ = baseLocData;
        }

        float *getBaseVideoData() const
        {
            return base_video_data_;
        }

        void setBaseVideoData(float *baseVideoData)
        {
            base_video_data_ = baseVideoData;
        }

        float *getQueryEmbData() const
        {
            return query_emb_data_;
        }

        void setQueryEmbData(float *queryEmbData)
        {
            query_emb_data_ = queryEmbData;
        }

        float *getQueryLocData() const
        {
            return query_loc_data_;
        }

        void setQueryLocData(float *queryLocData)
        {
            query_loc_data_ = queryLocData;
        }

        float *getQueryVideoData() const
        {
            return query_video_data_;
        }

        void setQueryVideoData(float *queryVideoData)
        {
            query_video_data_ = queryVideoData;
        }


        /************    ****************/
        float *getQueryWeightData() const
        {
            return query_alpha_;
        }

        void setQueryWeightData(float *queryWeightData)
        {
            query_alpha_ = queryWeightData;
        }

        unsigned int *getGroundData() const
        {
            return ground_data_;
        }

        void setGroundData(unsigned int *groundData)
        {
            ground_data_ = groundData;
        }

        unsigned int getBaseLen() const
        {
            return base_len_;
        }

        void setBaseLen(unsigned int baseLen)
        {
            base_len_ = baseLen;
        }

        unsigned int getQueryLen() const
        {
            return query_len_;
        }

        void setQueryLen(unsigned int queryLen)
        {
            query_len_ = queryLen;
        }

        unsigned int getGroundLen() const
        {
            return ground_len_;
        }

        void setGroundLen(unsigned int groundLen)
        {
            ground_len_ = groundLen;
        }

        unsigned int getBaseEmbDim() const
        {
            return base_emb_dim_;
        }

        unsigned int getBaseLocDim() const
        {
            return base_loc_dim_;
        }

        unsigned int getBaseVideoDim() const
        {
            return base_video_dim_;
        }
        
        void setBaseEmbDim(unsigned int baseEmbDim)
        {
            base_emb_dim_ = baseEmbDim;
            e_dist_->dist_para_set(baseEmbDim);
        }

        void setBaseLocDim(unsigned int baseLocDim)
        {
            base_loc_dim_ = baseLocDim;
            s_dist_->dist_para_set(baseLocDim);
        }

        void setBaseVideoDim(unsigned int baseVideoDim)
        {
            base_video_dim_ = baseVideoDim;
            v_dist_->dist_para_set(baseVideoDim);
        }

        unsigned int getQueryEmbDim() const
        {
            return query_emb_dim_;
        }

        unsigned int getQueryLocDim() const
        {
            return query_loc_dim_;
        }

        unsigned int getQueryVideoDim() const
        {
            return query_video_dim_;
        }

        void setQueryEmbDim(unsigned int queryEmbDim)
        {
            query_emb_dim_ = queryEmbDim;
        }

        void setQueryLocDim(unsigned int queryLocDim)
        {
            query_loc_dim_ = queryLocDim;
        }

        void setQueryVideoDim(unsigned int queryVideoDim)
        {
            query_video_dim_ = queryVideoDim;
        }

        unsigned int getGroundDim() const
        {
            return ground_dim_;
        }

        void setGroundDim(unsigned int groundDim)
        {
            ground_dim_ = groundDim;
        }

        Parameters &getParam()
        {
            return param_;
        }

        void setParam(const Parameters &param)
        {
            param_ = param;
        }

        unsigned int getInitEdgesNum() const
        {
            return init_edges_num;
        }

        void setInitEdgesNum(unsigned int initEdgesNum)
        {
            init_edges_num = initEdgesNum;
        }

        unsigned int getCandidatesEdgesNum() const
        {
            return candidates_edges_num;
        }

        unsigned int getUpdateLayerNum() const
        {
            return update_layer_num;
        }

        void setCandidatesEdgesNum(unsigned int candidatesEdgesNum)
        {
            candidates_edges_num = candidatesEdgesNum;
        }

        void setUpdateLayerNum(unsigned int updatelayernum)
        {
            update_layer_num = updatelayernum;
        }

        unsigned int getResultEdgesNum() const
        {
            return result_edges_num;
        }

        void setResultEdgesNum(unsigned int resultEdgesNum)
        {
            result_edges_num = resultEdgesNum;
        }

        void set_alpha(float alpha)
        {
            param_.set<float>("alpha", alpha);
        }

        void set_alpha_1(float alpha_1)
        {
            param_.set<float>("alpha_1", alpha_1);
        }

        void set_alpha_2(float alpha_2)
        {
            param_.set<float>("alpha_2", alpha_2);
        }

        float get_alpha_1() const
        {
            return param_.get<float>("alpha_1");
        }

        float get_alpha_2() const
        {
            return param_.get<float>("alpha_2");
        }

        float get_alpha() const
        {
            return param_.get<float>("alpha");
        }

        void set_alpha_index_1(unsigned int index_d1)
        {
            index_d1_ = index_d1;
        }

        unsigned int get_alpha_index_1() const
        {
            return index_d1_; 
        }

        void set_alpha_index_2(unsigned int index_d2)
        {
            index_d2_ = index_d2;
        }

        unsigned int get_alpha_index_2() const
        {
            return index_d2_; 
        }

        E_Distance *get_E_Dist() const
        {
            return e_dist_;
        }


        E_Distance *get_S_Dist() const
        {
            return s_dist_;
        }

        E_Distance *get_V_Dist() const
        {
            return v_dist_;
        }

        FinalGraph &getFinalGraph()
        {
            return final_graph_;
        }

        LoadGraph &getLoadGraph()
        {
            return load_graph_;
        }

        LoadGraph &getExactGraph()
        {
            return exact_graph_;
        }

        TYPE getCandidateType() const
        {
            return candidate_type;
        }

        void setCandidateType(TYPE candidateType)
        {
            candidate_type = candidateType;
        }

        TYPE getPruneType() const
        {
            return prune_type;
        }

        void setPruneType(TYPE pruneType)
        {
            prune_type = pruneType;
        }

        TYPE getEntryType() const
        {
            return entry_type;
        }

        void setEntryType(TYPE entryType)
        {
            entry_type = entryType;
        }

        void setConnType(TYPE connType)
        {
            conn_type = connType;
        }

        TYPE getConnType() const
        {
            return conn_type;
        }

        unsigned int getDistCount() const
        {
            return dist_count;
        }

        void resetDistCount()
        {
            dist_count = 0;
        }

        void addDistCount()
        {
            dist_count += 1;
        }

        unsigned int getHopCount() const
        {
            return hop_count;
        }

        void resetHopCount()
        {
            hop_count = 0;
        }

        void addHopCount()
        {
            hop_count += 1;
        }

        void setNumThreads(const unsigned numthreads)
        {
            omp_set_num_threads(numthreads);
        }

        int i = 0;
        bool debug = false;

    private:
        float *base_emb_data_, *base_loc_data_, *base_video_data_, *query_emb_data_, *query_loc_data_, *query_video_data_, *query_alpha_;  // 原始数据和query
        unsigned *ground_data_;  

        unsigned base_len_, query_len_, ground_len_; 
        unsigned base_emb_dim_, base_loc_dim_, base_video_dim_, query_emb_dim_, query_loc_dim_, query_video_dim_, ground_dim_;  // 一个向量的维度

        Parameters param_;
        unsigned init_edges_num;       // S
        unsigned candidates_edges_num; // L
        unsigned result_edges_num;     // K
        unsigned update_layer_num;

        E_Distance *e_dist_;  
        E_Distance *s_dist_;
        E_Distance *v_dist_;

        FinalGraph final_graph_;  
        LoadGraph load_graph_;
        LoadGraph exact_graph_;

        TYPE entry_type;
        TYPE candidate_type;
        TYPE prune_type;
        TYPE conn_type;

        unsigned dist_count = 0;
        unsigned hop_count = 0;

        unsigned index_d1_ = 0;
        unsigned index_d2_ = 0;

        float alpha_;
        float alpha1_;
        float alpha2_;
        float max_emb_dist_, max_spatial_dist_, max_video_dist_;
    };
}

#endif