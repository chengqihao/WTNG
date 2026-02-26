#include "parameters.h"
#include <string.h>
#include <iostream>


void HNSW_PARA(std::string dataset, stkq::Parameters &parameters)
{
    unsigned max_m, max_m0, ef_construction;
    if (dataset == "openimage")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }
    else if (dataset == "cc3m")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }

    else if (dataset == "mugen")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }
    else if (dataset == "imagenet")
    {
        max_m = 40, max_m0 = 40, ef_construction = 200;
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<int>("mult", -1);
    parameters.set<unsigned>("max_m", max_m);
    parameters.set<unsigned>("max_m0", max_m0);
    parameters.set<unsigned>("ef_construction", ef_construction);
}


void WTNG_PARA(std::string dataset, stkq::Parameters &parameters)
{
    unsigned max_m, ef_construction;
    if (dataset == "openimage")
    {
        max_m = 40, ef_construction = 200;
    }
    else if (dataset == "cc3m")
    {
        max_m = 40, ef_construction = 200;
    }
    else if (dataset == "mugen")
    {
        max_m = 40, ef_construction = 200;
    }
    else if (dataset == "imagenet")
    {
        max_m = 40, ef_construction = 200;
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("max_m", max_m);
    parameters.set<unsigned>("ef_construction", ef_construction);
    parameters.set<int>("mult", -1);
}


void set_data_path(std::string dataset, stkq::Parameters &parameters)
{
    std::string dataset_root = parameters.get<std::string>("dataset_root");
    std::string base_emb_path(dataset_root);
    std::string base_loc_path(dataset_root);
    std::string base_video_path(dataset_root);
    std::string query_emb_path(dataset_root);
    std::string query_loc_path(dataset_root);
    std::string query_video_path(dataset_root);
    std::string query_alpha_path(dataset_root);
    std::string partition_path(dataset_root);
    std::string ground_path(dataset_root);
    float alpha_1 = parameters.get<float>("alpha_1");
    float alpha_2 = parameters.get<float>("alpha_2");
    int range = 0;
    std::cout << alpha_1 << alpha_2 << std::endl;
    const float epsilon = 1e-6;


    if (std::fabs(alpha_1 - 0.7f) < epsilon && std::fabs(alpha_2 - 0.1f) < epsilon)
    {
        range = 0;
    }
    else if (std::fabs(alpha_1 - 0.1f) < epsilon && std::fabs(alpha_2 - 0.7f) < epsilon)
    {
        range = 1;
    }
    else if (std::fabs(alpha_1 - 0.1f) < epsilon && std::fabs(alpha_2 - 0.1f) < epsilon)
    {
        range = 2;
    }
    else if (std::fabs(alpha_1 - 0.05f) < epsilon && std::fabs(alpha_2 - 0.45f) < epsilon)
    {
        range = 3;
    }
    else if (std::fabs(alpha_1 - 0.45f) < epsilon && std::fabs(alpha_2 - 0.05f) < epsilon)
    {
        range = 4;
    }
    else if (std::fabs(alpha_1 - 0.45f) < epsilon && std::fabs(alpha_2 - 0.45f) < epsilon)
    {
        range = 5;
    }
    else if (std::fabs(alpha_1 - 0.3f) < epsilon && std::fabs(alpha_2 - 0.3f) < epsilon)
    {
        range = 6;
    }
    else
    {
        std::cout << "alpha input error!\n";
        exit(-1);
    }

    std::cout << "Range: " << range << std::endl;

    if (dataset == "openimage")
    {
        base_emb_path.append(R"(OpenImage/base_img_emb.fvecs)"); 
        base_loc_path.append(R"(OpenImage/base_text_emb.fvecs)");
        base_video_path.append(R"(OpenImage/base_video_emb.fvecs)");
        query_emb_path.append(R"(OpenImage/query_img_emb.fvecs)");
        query_loc_path.append(R"(OpenImage/query_text_emb.fvecs)");
        query_video_path.append(R"(OpenImage/query_video_emb.fvecs)");
        query_alpha_path.append(R"(OpenImage/range2d_)" + std::to_string(range) + "_query_alpha.fvecs");
        ground_path.append(R"(OpenImage/range2d_)" + std::to_string(range) + "_top10_results.ivecs");
    }
    else if (dataset == "cc3m")
    {
        base_emb_path.append(R"(CC3M/base_img_emb.fvecs)");
        base_loc_path.append(R"(CC3M/base_text_emb.fvecs)");
        base_video_path.append(R"(CC3M/base_video_emb.fvecs)");
        query_emb_path.append(R"(CC3M/query_img_emb.fvecs)");
        query_loc_path.append(R"(CC3M/query_text_emb.fvecs)");
        query_video_path.append(R"(CC3M/query_video_emb.fvecs)");
        query_alpha_path.append(R"(CC3M/range2d_)" + std::to_string(range) + "_query_alpha.fvecs");
        ground_path.append(R"(CC3M/range2d_)" + std::to_string(range) + "_top10_results.ivecs");
    }
    else if (dataset == "mugen")
    {
        base_emb_path.append(R"(Mugen/base_img_emb.fvecs)");
        base_loc_path.append(R"(Mugen//base_text_emb.fvecs)");
        base_video_path.append(R"(Mugen/base_video_emb.fvecs)");
        query_emb_path.append(R"(Mugen/query_img_emb.fvecs)");
        query_loc_path.append(R"(Mugen/query_text_emb.fvecs)");
        query_video_path.append(R"(Mugen/query_video_emb.fvecs)");
        query_alpha_path.append(R"(Mugen/range2d_)" + std::to_string(range) + "_query_alpha.fvecs");
        ground_path.append(R"(Mugen/range2d_)" + std::to_string(range) + "_top10_results.ivecs");
    }
    else if (dataset == "imagenet")
    {
        base_emb_path.append(R"(Imagenet/base_img_emb.fvecs)");
        base_loc_path.append(R"(Imagenet//base_text_emb.fvecs)");
        base_video_path.append(R"(Imagenet/base_video_emb.fvecs)");
        query_emb_path.append(R"(Imagenet/query_img_emb.fvecs)");
        query_loc_path.append(R"(Imagenet/query_text_emb.fvecs)");
        query_video_path.append(R"(Imagenet/query_video_emb.fvecs)");
        query_alpha_path.append(R"(Imagenet/range2d_)" + std::to_string(range) + "_query_alpha.fvecs");
        ground_path.append(R"(Imagenet/range2d_)" + std::to_string(range) + "_top10_results.ivecs");
    }
    else
    {
        std::cout << "dataset input error!\n";
        exit(-1);
    }
    parameters.set<std::string>("base_emb_path", base_emb_path);
    parameters.set<std::string>("base_loc_path", base_loc_path);
    parameters.set<std::string>("base_video_path", base_video_path);
    parameters.set<std::string>("query_emb_path", query_emb_path);
    parameters.set<std::string>("query_loc_path", query_loc_path);
    parameters.set<std::string>("query_video_path", query_video_path);
    parameters.set<std::string>("query_alpha_path", query_alpha_path);
    parameters.set<std::string>("ground_path", ground_path);
}

void set_para(std::string alg, std::string dataset, stkq::Parameters &parameters)
{
    set_data_path(dataset, parameters);
    if (parameters.get<std::string>("exc_type") != "build")
    {
        return;
    }

    if (alg == "hnsw")
    {
        HNSW_PARA(dataset, parameters);
    }
    else if (alg == "wtng")
    {
        WTNG_PARA(dataset, parameters);
    }
    else if (alg == "baseline1")
    {
        HNSW_PARA(dataset, parameters);
    }
    else if (alg == "baseline2")
    {
        HNSW_PARA(dataset, parameters);
    }
    else
    {
        std::cout << "algorithm input error!\n";
        exit(-1);
    }
}