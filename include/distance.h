#ifndef STKQ_DISTANCE_H
#define STKQ_DISTANCE_H

#include <immintrin.h>
#include <cmath>
#include "hnswlib/hnswalg.h"
namespace stkq
{

    class E_Distance
    {
    public:
        using dist_t = float;
        
        E_Distance(float max_emb_dist) : max_emb_dist(max_emb_dist),
            inv_max_emb_dist(max_emb_dist > 0 ? 1.0f / max_emb_dist : 0.0f),
            space_(nullptr),
            fstdistfunc_(nullptr),
            dist_func_param_(nullptr)
        {

        }
        
        E_Distance(const E_Distance&) = delete;
        E_Distance& operator=(const E_Distance&) = delete;

        ~E_Distance() {
            if (space_ != nullptr){
                delete space_;
            }
        }

        void dist_para_set(unsigned dim){
            delete space_;
            space_ = new hnswlib::L2Space(dim);
            fstdistfunc_     = space_->get_dist_func();
            dist_func_param_ = space_->get_dist_func_param();
        }

        template <typename T>
        T compare(const T* a, const T* b, unsigned length) const {
            dist_t raw = fstdistfunc_(
                a,
                b,
                dist_func_param_
            );

            return std::sqrt(raw) * inv_max_emb_dist;
        }

        template <typename T>
        T compare_square(const T* a, const T* b, unsigned length) const {
            dist_t raw = fstdistfunc_(
                a,
                b,
                dist_func_param_
            );

            return raw * inv_max_emb_dist * inv_max_emb_dist;
        }

    private:
        float max_emb_dist = 0;
        float inv_max_emb_dist = 0.f;
    public:
        hnswlib::SpaceInterface<dist_t>* space_;
        hnswlib::DISTFUNC<dist_t>        fstdistfunc_;
        void*                            dist_func_param_;
    };

    class S_Distance
    {
    public:
        template <typename T>
        T compare(const T *a, const T *b, unsigned length) const
        {
            T spatial_distance = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
            return std::sqrt(spatial_distance) / max_spatial_dist;
        }
        S_Distance(float max_spatial_dist) : max_spatial_dist(max_spatial_dist) {}

    private:
        float max_spatial_dist = 0;
    };
}

#endif
