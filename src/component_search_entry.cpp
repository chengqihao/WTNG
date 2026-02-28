//
// Created by MurphySL on 2020/10/23.
//

#include "component.h"

namespace stkq
{
    void ComponentSearchEntryCentroid::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool)
    {

        const auto L = index->getParam().get<unsigned>("L_search");  
        pool.reserve(L + 1);
        std::vector<unsigned> init_ids(L);  
        boost::dynamic_bitset<> flags{index->getBaseLen(), 0};  

        unsigned tmp_l = 0;
        for (; tmp_l < L && tmp_l < index->getLoadGraph()[index->ep_].size(); tmp_l++)
        {

            init_ids[tmp_l] = index->getLoadGraph()[index->ep_][tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L)
        {

            unsigned id = rand() % index->getBaseLen();
            if (flags[id])
                continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            
            unsigned id = init_ids[i];

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + (size_t)query * index->getQueryEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                     index->getBaseLocData() + (size_t)query * index->getBaseLocDim(),
                                                     index->getBaseLocDim());

            float v_d = index->get_V_Dist()->compare(index->getBaseVideoData() + (size_t)id * index->getBaseVideoDim(),
                                                     index->getBaseVideoData() + (size_t)query * index->getBaseVideoDim(),
                                                     index->getBaseVideoDim());           

            float dist = index->get_alpha_1() * e_d + index->get_alpha_2() * s_d + (1 - index->get_alpha_1() - index->get_alpha_2()) * v_d;

            index->addDistCount();
            pool[i] = Index::Neighbor(id, dist, true);
        }

        std::sort(pool.begin(), pool.begin() + L);
    }

    void ComponentSearchEntryNone::SearchEntryInner(unsigned int query, std::vector<Index::Neighbor> &pool) {}

}
