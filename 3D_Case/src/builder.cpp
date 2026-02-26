
#include "builder.h"
#include "component.h"
#include <set>

namespace stkq
{

    IndexBuilder *IndexBuilder::load(char *data_emb_file, char *data_loc_file, char *data_video_file, char *query_emb_file, char *query_loc_file, char *query_video_file, char *query_alpha_file, char *ground_file, Parameters &parameters, bool dual)
    {
        if (!dual)
        {
            auto *a = new ComponentLoad(final_index_);
            a->LoadInner(data_emb_file, data_loc_file, data_video_file, query_emb_file, query_loc_file, query_video_file, query_alpha_file, ground_file, parameters);
            std::cout << "base data len : " << final_index_->getBaseLen() << std::endl;
            std::cout << "base data emb dim : " << final_index_->getBaseEmbDim() << std::endl;
            std::cout << "base data loc dim : " << final_index_->getBaseLocDim() << std::endl;
            std::cout << "base data video dim : " << final_index_->getBaseVideoDim() << std::endl;
            std::cout << "query data len : " << final_index_->getQueryLen() << std::endl;
            std::cout << "query data emb dim : " << final_index_->getQueryEmbDim() << std::endl;
            std::cout << "query data loc dim : " << final_index_->getQueryLocDim() << std::endl;
            std::cout << "query data video dim : " << final_index_->getQueryVideoDim() << std::endl;
            std::cout << "ground truth data len : " << final_index_->getGroundLen() << std::endl;
            std::cout << "ground truth data dim : " << final_index_->getGroundDim() << std::endl;
            std::cout << "=====================" << std::endl;
            std::cout << final_index_->getParam().toString() << std::endl;
            std::cout << "=====================" << std::endl;
            return this;
        }
        else
        {
            auto *a = new ComponentLoad(final_index_1);
            a->LoadInner(data_emb_file, data_loc_file, data_video_file, query_emb_file, query_loc_file, query_video_file, query_alpha_file, ground_file, parameters);
            final_index_1->set_alpha_1(0);
            final_index_1->set_alpha_2(1);
            auto *b = new ComponentLoad(final_index_2);
            b->LoadInner(data_emb_file, data_loc_file, data_video_file, query_emb_file, query_loc_file, query_video_file, query_alpha_file, ground_file, parameters);
            final_index_2->set_alpha_1(1);
            final_index_2->set_alpha_2(0);
            auto *c = new ComponentLoad(final_index_3);
            c->LoadInner(data_emb_file, data_loc_file, data_video_file, query_emb_file, query_loc_file, query_video_file, query_alpha_file, ground_file, parameters);
            final_index_3->set_alpha_1(0);
            final_index_3->set_alpha_2(0); 
            std::cout << "base data len : " << final_index_1->getBaseLen() << std::endl;
            std::cout << "base data emb dim : " << final_index_1->getBaseEmbDim() << std::endl;
            std::cout << "base data loc dim : " << final_index_1->getBaseLocDim() << std::endl;
            std::cout << "base data video dim : " << final_index_1->getBaseVideoDim() << std::endl;
            std::cout << "query data len : " << final_index_1->getQueryLen() << std::endl;
            std::cout << "query data emb dim : " << final_index_1->getQueryEmbDim() << std::endl;
            std::cout << "query data loc dim : " << final_index_1->getQueryLocDim() << std::endl;
            std::cout << "query data video dim : " << final_index_1->getQueryVideoDim() << std::endl;
            std::cout << "ground truth data len : " << final_index_1->getGroundLen() << std::endl;
            std::cout << "ground truth data dim : " << final_index_1->getGroundDim() << std::endl;
            std::cout << "=====================" << std::endl;
            std::cout << final_index_1->getParam().toString() << std::endl;
            std::cout << final_index_2->getParam().toString() << std::endl;
            std::cout << final_index_3->getParam().toString() << std::endl;

            std::cout << final_index_1->get_alpha_1() << std::endl;
            std::cout << final_index_1->get_alpha_2() << std::endl;
            std::cout << final_index_2->get_alpha_1() << std::endl;
            std::cout << final_index_2->get_alpha_2() << std::endl;
            std::cout << final_index_3->get_alpha_1() << std::endl;
            std::cout << final_index_3->get_alpha_2() << std::endl;
            std::cout << "=====================" << std::endl;
            return this;
        }
    }

    IndexBuilder *IndexBuilder::init(TYPE type, bool debug)
    {
        s = std::chrono::high_resolution_clock::now();
        ComponentInit *a = nullptr;

        if (type == INIT_HNSW)
        {
            std::cout << "__INIT : HNSW__" << std::endl;
            a = new ComponentInitHNSW(final_index_);
        }
        else if (type == INIT_WTNG)
        {
            std::cout << "__INIT : WTNG__" << std::endl;
            a = new ComponentInitWTNG(final_index_);
        }
        else if (type == INIT_RANDOM)
        {
            std::cout << "__INIT : RANDOM__" << std::endl;
            a = new ComponentInitRandom(final_index_);
        }
        else
        {
            std::cerr << "__INIT : WRONG TYPE__" << std::endl;
            exit(-1);
        }
        std::cout << "before initinner" << std::endl;
        a->InitInner();
        e = std::chrono::high_resolution_clock::now();
        std::cout << "__INIT FINISH__" << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
        std::cout << "Initialization time: " << duration << " milliseconds" << std::endl;
        std::cout << "ef construction number:" << final_index_->WTNG_enterpoints.size() << std::endl;
        return this;
    }

    IndexBuilder *IndexBuilder::save_graph(TYPE type, char *graph_file)
    {

        std::fstream out(graph_file, std::ios::binary | std::ios::out);
        if (type == INDEX_HNSW)
        {
            unsigned enterpoint_id = final_index_->enterpoint_->GetId();
            unsigned max_level = final_index_->max_level_;
            out.write((char *)&enterpoint_id, sizeof(unsigned));
            out.write((char *)&max_level, sizeof(unsigned));
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id = final_index_->nodes_[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
                unsigned node_level = final_index_->nodes_[i]->GetLevel() + 1;
                out.write((char *)&node_level, sizeof(unsigned));
                unsigned current_level_GK;
                for (unsigned j = 0; j < node_level; j++)
                {
                    current_level_GK = final_index_->nodes_[i]->GetFriends(j).size();
                    out.write((char *)&current_level_GK, sizeof(unsigned));
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id = final_index_->nodes_[i]->GetFriends(j)[k]->GetId();
                        out.write((char *)&current_level_neighbor_id, sizeof(unsigned));
                    }
                }
            }
            out.close();
            return this;
        }
        else if (type == INDEX_WTNG)
        {
            int average_neighbor_size = 0;
            unsigned enterpoint_set_size = final_index_->WTNG_enterpoints.size();
            out.write((char *)&enterpoint_set_size, sizeof(unsigned));
            for (unsigned i = 0; i < enterpoint_set_size; i++)
            {
                unsigned node_id = final_index_->WTNG_enterpoints[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
            }

            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id = final_index_->WTNG_nodes_[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
                unsigned neighbor_size = final_index_->WTNG_nodes_[i]->GetFriends().size();
                out.write((char *)&neighbor_size, sizeof(unsigned));
                average_neighbor_size = average_neighbor_size + neighbor_size;

                for (unsigned k = 0; k < neighbor_size; k++)
                {
                    Index::WTNGNeighbor &neighbor = final_index_->WTNG_nodes_[i]->GetFriends()[k];
                    unsigned neighbor_id = neighbor.id_;
                    out.write((char *)&neighbor_id, sizeof(unsigned));

                    // TODO: 这个地方需要修改
                    const std::vector<uint8_t>& flags1d = neighbor.available_range_discrete;

                    // 写长度
                    uint32_t len = static_cast<uint32_t>(flags1d.size());
                    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
                    
                    // 写数据
                    if (len > 0) {
                        out.write(reinterpret_cast<const char*>(flags1d.data()),
                                  len * sizeof(uint8_t));
                    }
                                    
                }
            }
            out.close();
            return this;
        }

        for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
        {
            unsigned GK = (unsigned)final_index_->getFinalGraph()[i].size();
            std::vector<unsigned> tmp;
            for (unsigned j = 0; j < GK; j++)
            {
                tmp.push_back(final_index_->getFinalGraph()[i][j].id);
            }
            out.write((char *)&GK, sizeof(unsigned));
            out.write((char *)tmp.data(), GK * sizeof(unsigned));
        }
        out.close();

        // std::vector<std::vector<Index::SimpleNeighbor>>().swap(final_index_->getFinalGraph());
        return this;
    }

    IndexBuilder *IndexBuilder::load_graph(TYPE type, char *graph_file)
    {
        int average_neighbor_size = 0;
        int l1_average_neighbor_size = 0;

        std::ifstream in(graph_file, std::ios::binary);

        if (!in.is_open())
        {
            std::cerr << "load graph error: " << graph_file << std::endl;
            exit(-1);
        }
        if (type == INDEX_HNSW)
        {
            final_index_->nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            unsigned enterpoint_id;
            in.read((char *)&enterpoint_id, sizeof(unsigned));
            in.read((char *)&final_index_->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in.read((char *)&node_id, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetId(node_id);
                in.read((char *)&node_level, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetLevel(node_level);

                for (unsigned j = 0; j < node_level; j++)
                {
                    in.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        // final_index_->nodes_[current_level_neighbor_id]->SetId(current_level_neighbor_id);
                        tmp.push_back(final_index_->nodes_[current_level_neighbor_id]);
                    }
                    final_index_->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_->getBaseLen() << std::endl;
            final_index_->enterpoint_ = final_index_->nodes_[enterpoint_id];
            return this;
        }
        else if (type == INDEX_WTNG)
        {
            int average_neighbor_size = 0;
            final_index_->WTNG_nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->WTNG_nodes_[i] = new stkq::WTNG::WTNGNode(0, 0);
            }
            unsigned enterpoint_id, enterpoint_size;
            final_index_->enterpoint_set.clear();
            in.read((char *)&enterpoint_size, sizeof(unsigned));
            for (unsigned i = 0; i < enterpoint_size; i++)
            {
                in.read((char *)&enterpoint_id, sizeof(unsigned));
                final_index_->enterpoint_set.push_back(enterpoint_id);
            }

            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id, neighbor_size;
                in.read((char *)&node_id, sizeof(unsigned));
                final_index_->WTNG_nodes_[i]->SetId(node_id);
                in.read((char *)&neighbor_size, sizeof(unsigned));
                average_neighbor_size = average_neighbor_size + neighbor_size;
                final_index_->WTNG_nodes_[i]->SetMaxM(neighbor_size);
                std::vector<Index::WTNGSimpleNeighbor> neighbors;
                neighbors.reserve(neighbor_size);
                int max_layer = 0;
                for (unsigned k = 0; k < neighbor_size; k++)
                {
                    unsigned neighbor_id;
                    in.read((char *)&neighbor_id, sizeof(unsigned));

                    // <<< 在这里加 >>>
                    uint32_t len = 0;
                    in.read(reinterpret_cast<char*>(&len), sizeof(len));
                    
                    std::vector<uint8_t> flags1d(len);
                    if (len > 0) {
                        in.read(reinterpret_cast<char*>(flags1d.data()), len * sizeof(uint8_t));
                    }
                    
                    // 注意：你的构造函数也要改成接收 vector<uint8_t>
                    neighbors.push_back(Index::WTNGSimpleNeighbor(neighbor_id, flags1d));
                }
                final_index_->WTNG_nodes_[i]->SetSearchFriends(neighbors);
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_->getBaseLen() << std::endl;
            return this;
        }

        while (!in.eof())
        {
            unsigned GK;
            in.read((char *)&GK, sizeof(unsigned));
            if (in.eof())
                break;
            std::vector<unsigned> tmp(GK);
            in.read((char *)tmp.data(), GK * sizeof(unsigned));
            final_index_->getLoadGraph().push_back(tmp);
        }
        return this;
    }

    IndexBuilder *IndexBuilder::load_graph(TYPE type, char *graph_file_1, char *graph_file_2)
    {
        std::ifstream in1(graph_file_1, std::ios::binary);
        std::ifstream in2(graph_file_2, std::ios::binary);

        int average_neighbor_size = 0;
        int l1_average_neighbor_size = 0;

        if (!in1.is_open())
        {
            std::cerr << "load graph error: " << graph_file_1 << std::endl;
            exit(-1);
        }

        if (!in2.is_open())
        {
            std::cerr << "load graph error: " << graph_file_2 << std::endl;
            exit(-1);
        }

        if (type == INDEX_HNSW)
        {
            final_index_1->nodes_.resize(final_index_1->getBaseLen());
            for (unsigned i = 0; i < final_index_1->getBaseLen(); i++)
            {
                final_index_1->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            unsigned enterpoint_id;
            in1.read((char *)&enterpoint_id, sizeof(unsigned));
            in1.read((char *)&final_index_1->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_1->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in1.read((char *)&node_id, sizeof(unsigned));
                final_index_1->nodes_[node_id]->SetId(node_id);
                in1.read((char *)&node_level, sizeof(unsigned));
                final_index_1->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in1.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in1.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_1->nodes_[current_level_neighbor_id]);
                    }
                    final_index_1->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_1->getBaseLen() << std::endl;
            final_index_1->enterpoint_ = final_index_1->nodes_[enterpoint_id];

            average_neighbor_size = 0;
            l1_average_neighbor_size = 0;
            final_index_2->nodes_.resize(final_index_2->getBaseLen());
            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                final_index_2->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            in2.read((char *)&enterpoint_id, sizeof(unsigned));
            in2.read((char *)&final_index_2->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in2.read((char *)&node_id, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetId(node_id);
                in2.read((char *)&node_level, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in2.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in2.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_2->nodes_[current_level_neighbor_id]);
                    }
                    final_index_2->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_2->getBaseLen() << std::endl;
            final_index_2->enterpoint_ = final_index_2->nodes_[enterpoint_id];
        }
        else
        {
            std::cout << "error for index type" << std::endl;
            exit(1);
        }

        return this;
    }
    
    IndexBuilder *IndexBuilder::load_graph(TYPE type, char *graph_file_1, char *graph_file_2, char *graph_file_3)
    {
        std::ifstream in1(graph_file_1, std::ios::binary);
        std::ifstream in2(graph_file_2, std::ios::binary);
        std::ifstream in3(graph_file_3, std::ios::binary);

        int average_neighbor_size = 0;
        int l1_average_neighbor_size = 0;

        if (!in1.is_open())
        {
            std::cerr << "load graph error: " << graph_file_1 << std::endl;
            exit(-1);
        }

        if (!in2.is_open())
        {
            std::cerr << "load graph error: " << graph_file_2 << std::endl;
            exit(-1);
        }

        if (!in3.is_open())
        {
            std::cerr << "load graph error: " << graph_file_3 << std::endl;
            exit(-1);
        }

        if (type == INDEX_HNSW)
        {
            final_index_1->nodes_.resize(final_index_1->getBaseLen());
            for (unsigned i = 0; i < final_index_1->getBaseLen(); i++)
            {
                final_index_1->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            unsigned enterpoint_id;
            in1.read((char *)&enterpoint_id, sizeof(unsigned));
            in1.read((char *)&final_index_1->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_1->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in1.read((char *)&node_id, sizeof(unsigned));
                final_index_1->nodes_[node_id]->SetId(node_id);
                in1.read((char *)&node_level, sizeof(unsigned));
                final_index_1->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in1.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in1.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_1->nodes_[current_level_neighbor_id]);
                    }
                    final_index_1->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_1->getBaseLen() << std::endl;
            final_index_1->enterpoint_ = final_index_1->nodes_[enterpoint_id];

            average_neighbor_size = 0;
            l1_average_neighbor_size = 0;

            final_index_2->nodes_.resize(final_index_2->getBaseLen());
            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                final_index_2->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            in2.read((char *)&enterpoint_id, sizeof(unsigned));
            in2.read((char *)&final_index_2->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_2->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in2.read((char *)&node_id, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetId(node_id);
                in2.read((char *)&node_level, sizeof(unsigned));
                final_index_2->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in2.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in2.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_2->nodes_[current_level_neighbor_id]);
                    }
                    final_index_2->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_2->getBaseLen() << std::endl;
            final_index_2->enterpoint_ = final_index_2->nodes_[enterpoint_id];

            average_neighbor_size = 0;
            l1_average_neighbor_size = 0;

            final_index_3->nodes_.resize(final_index_3->getBaseLen());
            for (unsigned i = 0; i < final_index_3->getBaseLen(); i++)
            {
                final_index_3->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            in3.read((char *)&enterpoint_id, sizeof(unsigned));
            in3.read((char *)&final_index_3->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_3->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in3.read((char *)&node_id, sizeof(unsigned));
                final_index_3->nodes_[node_id]->SetId(node_id);
                in3.read((char *)&node_level, sizeof(unsigned));
                final_index_3->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in3.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in3.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(final_index_3->nodes_[current_level_neighbor_id]);
                    }
                    final_index_3->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_3->getBaseLen() << std::endl;
            final_index_3->enterpoint_ = final_index_3->nodes_[enterpoint_id];
        }
        else
        {
            std::cout << "error for index type" << std::endl;
            exit(1);
        }

        return this;
    }
    
    /**
     * offline search
     * param entry_type
     * param route_type
     * return
     */
    IndexBuilder *IndexBuilder::search(TYPE entry_type, TYPE route_type, TYPE L_type, Parameters param_)
    {
        std::cout << "__SEARCH__" << std::endl;

        unsigned K = 10; // 在近邻搜索中要找到的最近邻的数量, 迭代增加

        if (route_type == DUAL_ROUTER_HNSW)
        {
            final_index_1->getParam().set<unsigned>("K_search", K);
            final_index_2->getParam().set<unsigned>("K_search", K);
            final_index_3->getParam().set<unsigned>("K_search", K);
            std::vector<std::vector<unsigned>> res_1;
            std::vector<std::vector<unsigned>> res_2;
            std::vector<std::vector<unsigned>> res_3;
            std::cout << "__ROUTER : DUAL_HNSW__" << std::endl;
            ComponentSearchEntry *a1 = new ComponentSearchEntryNone(final_index_1);
            ComponentSearchEntry *a2 = new ComponentSearchEntryNone(final_index_2);
            ComponentSearchEntry *a3 = new ComponentSearchEntryNone(final_index_3);
            ComponentSearchRoute *b1 = new ComponentSearchRouteHNSW(final_index_1);
            ComponentSearchRoute *b2 = new ComponentSearchRouteHNSW(final_index_2);
            ComponentSearchRoute *b3 = new ComponentSearchRouteHNSW(final_index_3);

            if (L_type == L_SEARCH_ASCEND)
            {
                std::set<unsigned> visited;
                unsigned sg = 1000;
                float acc_set = 0.99;
                bool flag = false;
                int L_sl = 1;
                unsigned L = 0;
                unsigned k_plus = 0;
                visited.insert(L);
                unsigned L_min = 0x7fffffff;
                float alpha_1 = param_.get<float>("alpha_1");
                float alpha_2 = param_.get<float>("alpha_2");
                for (unsigned t = 0; t < 20; t++)
                {
                    L = L + K;
                    final_index_1->getParam().set<unsigned>("K_search", L);
                    final_index_2->getParam().set<unsigned>("K_search", L);
                    final_index_3->getParam().set<unsigned>("K_search", L);
                    std::cout << "SEARCH_L : " << L << std::endl;
                    if (L < K)
                    {
                        std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                        exit(-1);
                    }

                    final_index_1->getParam().set<unsigned>("L_search", L);
                    final_index_2->getParam().set<unsigned>("L_search", L);
                    final_index_3->getParam().set<unsigned>("L_search", L);

                    auto s1 = std::chrono::high_resolution_clock::now();

                    res_1.clear();
                    res_1.resize(final_index_1->getQueryLen());
                    for (unsigned i = 0; i < final_index_1->getQueryLen(); i++)
                    {
                        std::vector<Index::Neighbor> pool;
                        a1->SearchEntryInner(i, pool);
                        b1->RouteInner(i, pool, res_1[i]);
                    }

                    res_2.clear();
                    res_2.resize(final_index_2->getQueryLen());
                    for (unsigned i = 0; i < final_index_2->getQueryLen(); i++)
                    {
                        std::vector<Index::Neighbor> pool;
                        a2->SearchEntryInner(i, pool);
                        b2->RouteInner(i, pool, res_2[i]);
                    }

                    res_3.clear();
                    res_3.resize(final_index_3->getQueryLen());
                    for (unsigned i = 0; i < final_index_3->getQueryLen(); i++)
                    {
                        std::vector<Index::Neighbor> pool;
                        a3->SearchEntryInner(i, pool);
                        b3->RouteInner(i, pool, res_3[i]);
                    }

                    std::priority_queue<Index::CloserFirst> result_queue;
                    std::vector<std::vector<unsigned>> res;

                    for (int i = 0; i < res_1.size(); i++)
                    {
                        for (int j = 0; j < res_1[i].size(); j++)
                        {
                            float e_d = final_index_1->get_E_Dist()->compare(final_index_1->getQueryEmbData() + i * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbData() + res_1[i][j] * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbDim());

                            float s_d = final_index_1->get_S_Dist()->compare(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocData() + res_1[i][j] * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocDim());

                            float v_d = final_index_1->get_V_Dist()->compare(final_index_1->getQueryVideoData() + i * final_index_1->getBaseVideoDim(),
                                                                             final_index_1->getBaseVideoData() + res_1[i][j] * final_index_1->getBaseVideoDim(),
                                                                             final_index_1->getBaseVideoDim());

                            float d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1 - alpha_2) * v_d;

                            result_queue.emplace(final_index_1->nodes_[res_1[i][j]], d);
                        }

                        for (int j = 0; j < res_2[i].size(); j++)
                        {
                            float e_d = final_index_1->get_E_Dist()->compare(final_index_1->getQueryEmbData() + i * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbData() + res_2[i][j] * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbDim());

                            float s_d = final_index_1->get_S_Dist()->compare(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocData() + res_2[i][j] * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocDim());

                            float v_d = final_index_1->get_V_Dist()->compare(final_index_1->getQueryVideoData() + i * final_index_1->getBaseVideoDim(),
                                                                             final_index_1->getBaseVideoData() + res_2[i][j] * final_index_1->getBaseVideoDim(),
                                                                             final_index_1->getBaseVideoDim());

                            float d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1 - alpha_2) * v_d;

                            result_queue.emplace(final_index_1->nodes_[res_2[i][j]], d);
                        }

                        for (int j = 0; j < res_3[i].size(); j++)
                        {
                            float e_d = final_index_1->get_E_Dist()->compare(final_index_1->getQueryEmbData() + i * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbData() + res_3[i][j] * final_index_1->getBaseEmbDim(),
                                                                             final_index_1->getBaseEmbDim());

                            float s_d = final_index_1->get_S_Dist()->compare(final_index_1->getQueryLocData() + i * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocData() + res_3[i][j] * final_index_1->getBaseLocDim(),
                                                                             final_index_1->getBaseLocDim());

                            float v_d = final_index_1->get_V_Dist()->compare(final_index_1->getQueryVideoData() + i * final_index_1->getBaseVideoDim(),
                                                                             final_index_1->getBaseVideoData() + res_3[i][j] * final_index_1->getBaseVideoDim(),
                                                                             final_index_1->getBaseVideoDim());

                            float d = alpha_1 * e_d + alpha_2 * s_d + (1 - alpha_1 - alpha_2) * v_d;

                            result_queue.emplace(final_index_1->nodes_[res_3[i][j]], d);
                        }
                        
                        std::vector<unsigned> tmp_res;
                        // std::unordered_set<unsigned> unique_results;
                        while (!result_queue.empty())
                        {
                            int top_node_id = result_queue.top().GetNode()->GetId();
                            // if (tmp_res.size() < K && unique_results.find(top_node_id) == unique_results.end())
                            if (tmp_res.size() < K)
                            {
                                tmp_res.push_back(top_node_id);
                                // unique_results.insert(top_node_id);
                            }
                            result_queue.pop();
                        }
                        res.push_back(tmp_res);
                    }

                    auto e1 = std::chrono::high_resolution_clock::now();
                    // auto e2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = e1 - s1;
                    std::cout << "search time: " << diff.count() / final_index_1->getQueryLen() << "\n";

                    float recall = 0;

                    for (unsigned i = 0; i < final_index_2->getQueryLen(); i++)
                    {
                        // if (res_1[i].size() == 0 or res_2[i].size() == 0)
                        if (res[i].size() == 0)
                            continue;
                        float tmp_recall = 0;
                        float cnt = 0;

                        for (unsigned j = 0; j < K; j++)
                        {
                            unsigned k = 0;
                            for (; k < K; k++)
                            {
                                if (res[i][k] == final_index_2->getGroundData()[i * final_index_2->getGroundDim() + j])
                                    break;
                            }
                            if (k == K)
                                cnt++;
                        }
                        tmp_recall = (float)(K - cnt) / (float)K;
                        recall = recall + tmp_recall;
                    }
                    // float acc = 1 - (float)cnt / (final_index_->getGroundLen() * K);
                    float acc = recall / final_index_2->getQueryLen();
                    std::cout << K << " NN accuracy: " << acc << std::endl;
                }
            }
            e = std::chrono::high_resolution_clock::now();
            std::cout << "__SEARCH FINISH__" << std::endl;

            return this;
        }

        final_index_->getParam().set<unsigned>("K_search", K); 
        // 这行代码设置了索引的参数K_search为K的值. 这意味着在后续的搜索中, 将寻找每个查询点的10个最近邻

        std::vector<std::vector<unsigned>> res;

        // ENTRY
        ComponentSearchEntry *a = nullptr;
        if (entry_type == SEARCH_ENTRY_NONE)
        {
            std::cout << "__SEARCH ENTRY : NONE__" << std::endl;
            a = new ComponentSearchEntryNone(final_index_);
        }
        else if (entry_type == SEARCH_ENTRY_CENTROID)
        {
            std::cout << "__SEARCH ENTRY : CENTROID__" << std::endl;
            a = new ComponentSearchEntryCentroid(final_index_);
        }
        else
        {
            std::cerr << "__SEARCH ENTRY : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        // ROUTE
        ComponentSearchRoute *b = nullptr;
        if (route_type == ROUTER_GREEDY)
        {
            std::cout << "__ROUTER : GREEDY__" << std::endl;
            b = new ComponentSearchRouteGreedy(final_index_);
        }
        else if (route_type == ROUTER_HNSW)
        {
            std::cout << "__ROUTER : HNSW__" << std::endl;
            b = new ComponentSearchRouteHNSW(final_index_);
        }
        else if (route_type == ROUTER_WTNG)
        {
            std::cout << "__ROUTER : WTNG__" << std::endl;
            b = new ComponentSearchRouteWTNG(final_index_);
        }
        else
        {
            std::cerr << "__ROUTER : WRONG TYPE__" << std::endl;
            exit(-1);
        }
        // std::cout << final_index_->alpha << std::endl;

        if (L_type == L_SEARCH_ASCEND)  // 只有一种search 算法
        {
            std::set<unsigned> visited;
            unsigned sg = 1000;  // not used
            float acc_set = 0.9;  // not used
            bool flag = false;  // not used
            int L_sl = 1;  // not used 
            unsigned L = 0;  // not used 
            visited.insert(L);
            unsigned L_min = 0x7fffffff;

            for (unsigned t = 0; t < 20; t++)
            {

                L = L + K;  // L: 候选搜索邻居,  K:返回的邻居数目
                std::cout << "SEARCH_L : " << L << std::endl;
                if (L < K) // 这个判断没什么用
                {
                    std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                    exit(-1);
                }

                final_index_->getParam().set<unsigned>("L_search", L);

                auto s1 = std::chrono::high_resolution_clock::now();

                res.clear();
                res.resize(final_index_->getQueryLen());
                //  #pragma omp parallel for

                const int STRIDE = 3;

                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                //                for (unsigned i = 0; i < 1000; i++)
                {
                    float alpha_1 = final_index_->getQueryWeightData()[i * STRIDE + 0];
                    float alpha_2 = final_index_->getQueryWeightData()[i * STRIDE + 1];
                    // float alpha_3 = final_index_->getQueryWeightData()[i * STRIDE + 2];

                    final_index_->set_alpha_1(alpha_1);
                    final_index_->set_alpha_2(alpha_2);

                    // final_index_->set_alpha(final_index_->getQueryWeightData()[i]);
                    std::vector<Index::Neighbor> pool;
                    // searchEntryInner: 对于EntryNone,这里返回空集; 对于EntryCentroid,
                    a->SearchEntryInner(i, pool); // ComponentSearchEntryNone, 此处不执行, center会执行
                    b->RouteInner(i, pool, res[i]);
                }

                /*********    开销计算   **********/
                auto e1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e1 - s1;
                std::cout << "search time: " << diff.count() / final_index_->getQueryLen() << "\n";
                std::cout << "DistCount: " << final_index_->getDistCount() << std::endl;
                std::cout << "HopCount: " << final_index_->getHopCount() << std::endl;
                final_index_->resetDistCount();
                final_index_->resetHopCount();
                // int cnt = 0;
                /*******     计算recall     ********/
                float recall = 0;
                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                {
                    if (res[i].size() == 0)
                        continue;
                    float tmp_recall = 0;
                    float cnt = 0;
                    for (unsigned j = 0; j < K; j++)
                    {
                        unsigned k = 0;
                        for (; k < K; k++)
                        {
                            if (res[i][j] == final_index_->getGroundData()[i * final_index_->getGroundDim() + k])
                                break;
                        }
                        if (k == K)
                            cnt++;
                    }
                    tmp_recall = (float)(K - cnt) / (float)K;
                    recall = recall + tmp_recall;
                }
                // float acc = 1 - (float)cnt / (final_index_->getGroundLen() * K);
                float acc = recall / final_index_->getQueryLen();
                std::cout << K << " NN accuracy: " << acc << std::endl;
            }
        }
        e = std::chrono::high_resolution_clock::now();
        std::cout << "__SEARCH FINISH__" << std::endl;

        return this;
    }

    void IndexBuilder::peak_memory_footprint()
    {
        unsigned iPid = (unsigned)getpid();

        std::cout << "PID: " << iPid << std::endl;

        std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
        std::ifstream info(status_file);
        if (!info.is_open())
        {
            std::cout << "memory information open error!" << std::endl;
        }
        std::string tmp;
        while (getline(info, tmp))
        {
            if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
                std::cout << tmp << std::endl;
        }
        info.close();
    }

}