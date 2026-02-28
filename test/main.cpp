#include <builder.h>
#include <set_para.h>
#include <iostream>
#include <array>

void HNSW(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string base_video_path = parameters.get<std::string>("base_video_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_video_path = parameters.get<std::string>("query_video_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), parameters.get<float>("max_video_distance"));

    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        builder->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }
    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->load_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0])
            ->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_HNSW, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void baseline1(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string base_video_path = parameters.get<std::string>("base_video_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_video_path = parameters.get<std::string>("query_video_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), parameters.get<float>("max_video_distance"));

    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        parameters.set<float>("alpha_1", 0.33);
        parameters.set<float>("alpha_2", 0.33);
        builder->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0],&query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }
    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters);
        builder->peak_memory_footprint();
        builder->load_graph(stkq::TYPE::INDEX_HNSW, &graph_file[0]);
        builder->peak_memory_footprint();
        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_HNSW, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

void baseline2(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string base_video_path = parameters.get<std::string>("base_video_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_video_path = parameters.get<std::string>("query_video_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), parameters.get<float>("max_video_distance"), true);
    builder->set_begin_time();
    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        auto *builder_1 = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), parameters.get<float>("max_video_distance"));
        std::string graph_file_1 = graph_file + "_1";
        parameters.set<float>("alpha_1", 0);
        parameters.set<float>("alpha_2", 1);
        builder_1->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file_1[0]);

        auto *builder_2 = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), parameters.get<float>("max_video_distance"));

        std::string graph_file_2 = graph_file + "_2";
        parameters.set<float>("alpha_1", 1);
        parameters.set<float>("alpha_2", 0);
        builder_2->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0],&query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file_2[0]);
        builder->set_end_time();

        auto *builder_3 = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), parameters.get<float>("max_video_distance"));

        std::string graph_file_3 = graph_file + "_3";
        parameters.set<float>("alpha_1", 0);
        parameters.set<float>("alpha_2", 0);
        builder_3->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_HNSW)
            ->save_graph(stkq::TYPE::INDEX_HNSW, &graph_file_3[0]);
        builder->set_end_time();
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }
    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        std::string graph_file_1 = graph_file + "_1";
        std::string graph_file_2 = graph_file + "_2";
        std::string graph_file_3 = graph_file + "_3";
        builder->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters, true);
        builder->peak_memory_footprint();

        builder->load_graph(stkq::TYPE::INDEX_HNSW, &graph_file_1[0], &graph_file_2[0], &graph_file_3[0]);
        builder->peak_memory_footprint();

        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::DUAL_ROUTER_HNSW, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}


void wtng(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string base_video_path = parameters.get<std::string>("base_video_path");
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_video_path = parameters.get<std::string>("query_video_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"), parameters.get<float>("max_video_distance"));


    std::vector<float> alpha_set_1 = {0.83f, 0.5f, 0.17f, 0.83f, 0.5f, 0.17f, 0, 0, 0, 0.33};
    std::vector<float> alpha_set_2 = {0.17f, 0.5f, 0.83f, 0, 0, 0, 0.83f, 0.5f, 0.17f, 0.33};

    parameters.set("alpha_set_1", alpha_set_1);
    parameters.set("alpha_set_2", alpha_set_2);

    if (parameters.get<std::string>("exc_type") == "build")
    {
        // build
        builder->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters)
            ->init(stkq::INIT_WTNG)
            ->save_graph(stkq::TYPE::INDEX_WTNG, &graph_file[0]);
        std::cout << "Build cost: " << builder->GetBuildTime().count() << "s" << std::endl;
    }

    else if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        builder->load(&base_emb_path[0], &base_loc_path[0], &base_video_path[0], &query_emb_path[0], &query_loc_path[0], &query_video_path[0], &query_alpha_path[0], &ground_path[0], parameters);
        builder->peak_memory_footprint();
        builder->load_graph(stkq::TYPE::INDEX_WTNG, &graph_file[0]);
        builder->peak_memory_footprint();
        builder->search(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_WTNG, stkq::TYPE::L_SEARCH_ASCEND, parameters);
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}


int main(int argc, char **argv)
{
    if (argc != 9)
    {
        std::cout << "./main algorithm dataset alpha1 alpha2 maximum_spatial_distance maximum_emb_distance maximum_video_distance exc_type"
                  << std::endl;
        exit(-1);
    }

    stkq::Parameters parameters;
    // std::string dataset_root = R"(/home/hrtang/vector_db/DEG_3d/database/)";
    // std::string index_path = R"(/home/hrtang/vector_db/DEG_3d/saved_index/ver1/)";
    std::string dataset_root = R"(/home/cqh22@mails.tsinghua.edu.cn/hrtang/database3/)";
    std::string index_path = R"(/home/cqh22@mails.tsinghua.edu.cn/hrtang/DEG_3D_v6/save_index/)";
    parameters.set<std::string>("dataset_root", dataset_root);
    parameters.set<std::string>("index_path", index_path);
    parameters.set<unsigned>("n_threads", 16);

    std::string alg(argv[1]);
    std::string dataset(argv[2]);
    std::string alpha_1(argv[3]);
    std::string alpha_2(argv[4]);
    std::string maximum_spatial_distance(argv[5]);
    std::string maximum_emb_distance(argv[6]);
    std::string maximum_video_distance(argv[7]);
    std::string exc_type(argv[8]);

    parameters.set<float>("alpha_1", std::stof(alpha_1));
    parameters.set<float>("alpha_2", std::stof(alpha_2));

    parameters.set<float>("max_spatial_distance", std::stof(maximum_spatial_distance));
    parameters.set<float>("max_emb_distance", std::stof(maximum_emb_distance));
    parameters.set<float>("max_video_distance", std::stof(maximum_video_distance));

    std::cout << "algorithm: " << alg << std::endl;
    std::cout << "dataset: " << dataset << std::endl;
    std::cout << "alpha_1: " << alpha_1 << std::endl;
    std::cout << "alpha_2: " << alpha_2 << std::endl;
    std::cout << "max_emb_distance: " << maximum_emb_distance << std::endl;
    std::cout << "max_spatial_distance: " << maximum_spatial_distance << std::endl;
    std::cout << "max_video_distance: " << maximum_video_distance << std::endl;
    std::string graph_file(alg + "_" + dataset + ".index");
    parameters.set<std::string>("graph_file", index_path + graph_file);
    parameters.set<std::string>("exc_type", exc_type);
    set_para(alg, dataset, parameters);

    if (alg == "baseline1")
    {
        baseline1(parameters);
    }
    else if (alg == "baseline2")
    {
        baseline2(parameters);
    }
    else if (alg == "wtng")
    {
        wtng(parameters);
    }
    else
    {
        std::cout << "alg input error!\n";
        exit(-1);
    }
    return 0;
}