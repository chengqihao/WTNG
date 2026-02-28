#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include "component.h"

namespace stkq
{

    template <typename T>
    inline void load_data(const char *filename, T *&data, unsigned &num, unsigned &dim)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open())
        {
            std::cerr << "Error opening file " << filename << std::endl;
            exit(-1);
        }


        in.read((char *)&dim, 4);
        if (in.fail())
        {
            std::cerr << "Error reading dimension from file " << filename << std::endl;
            exit(-1);
        }

        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        auto f_size = (size_t)ss;

        num = (unsigned)(f_size / (dim + 1) / 4);

        size_t total_size = (size_t)num * dim;
        try
        {
            data = new T[total_size];
        }
        catch (std::bad_alloc &)
        {
            std::cerr << "Memory allocation failed for data in " << filename << std::endl;
            exit(-1);
        }

        in.seekg(0, std::ios::beg);

        const size_t block_size = 10000 * dim; 
        size_t offset = 0;

        while (offset < total_size)
        {
            size_t remaining = total_size - offset;
            size_t current_block_size = std::min(block_size, remaining);

            for (size_t i = 0; i < current_block_size / dim; ++i)
            {
                unsigned current_dim;
                in.read(reinterpret_cast<char *>(&current_dim), sizeof(current_dim));
                if (in.fail() || current_dim != dim)
                {
                    std::cerr << "Error reading dimension or dimension mismatch in file " << filename << " at index " << (offset / dim + i) << std::endl;
                    delete[] data;
                    exit(-1);
                }

                in.read(reinterpret_cast<char *>(data + offset + i * dim), dim * sizeof(T));
                if (in.fail())
                {
                    std::cerr << "Error reading data from file " << filename << " at index " << (offset / dim + i) << std::endl;
                    delete[] data;
                    exit(-1);
                }
            }

            offset += current_block_size;
        }

        in.close();

        std::cout << "Loaded " << num << " entries from " << filename << " with dimension " << dim << std::endl;
    }

    void ComponentLoad::LoadInner(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *query_alpha_file, char *ground_file,
                                  Parameters &parameters)
    {
        float *data_emb = nullptr;
        unsigned n{};
        unsigned emb_dim{};
        load_data<float>(data_emb_file, data_emb, n, emb_dim);
        index->setBaseEmbData(data_emb);
        index->setBaseLen(n);
        index->setBaseEmbDim(emb_dim);
        assert(index->getBaseEmbData() != nullptr && index->getBaseLen() != 0 && index->getBaseEmbDim() != 0);
        float *data_loc = nullptr;
        unsigned loc_n{};
        unsigned loc_dim{};
        load_data<float>(data_loc_file, data_loc, loc_n, loc_dim);
        index->setBaseLocData(data_loc);
        index->setBaseLocDim(loc_dim);
        assert(index->getBaseLocData() != nullptr && loc_n == index->getBaseLen());
        float *query_emb = nullptr;
        unsigned query_num{};
        unsigned query_emb_dim{};
        load_data<float>(query_emb_file, query_emb, query_num, query_emb_dim);
        index->setQueryEmbData(query_emb);
        index->setQueryLen(query_num);
        index->setQueryEmbDim(query_emb_dim);
        assert(index->getQueryEmbData() != nullptr && index->getQueryLen() != 0 && index->getQueryEmbDim() != 0);
        assert(index->getBaseEmbDim() == index->getQueryEmbDim());
        float *query_loc = nullptr;
        unsigned query_loc_num{};
        unsigned query_loc_dim{};
        load_data(query_loc_file, query_loc, query_loc_num, query_loc_dim);
        index->setQueryLocData(query_loc);
        index->setQueryLocDim(query_loc_dim);
        assert(query_loc_num == index->getQueryLen() && query_loc_dim == index->getBaseLocDim());
        float *query_alpha = nullptr;
        unsigned query_alpha_num{};
        unsigned query_alpha_dim{};
        load_data(query_alpha_file, query_alpha, query_alpha_num, query_alpha_dim);
        index->setQueryWeightData(query_alpha);
        assert(query_loc_num == index->getQueryLen());
        unsigned *ground_data = nullptr;
        unsigned ground_num{};
        unsigned ground_dim{};
        load_data<unsigned>(ground_file, ground_data, ground_num, ground_dim);
        index->setGroundData(ground_data);
        index->setGroundLen(ground_num);
        index->setGroundDim(ground_dim);
        assert(index->getGroundData() != nullptr && index->getGroundLen() != 0 && index->getGroundDim() != 0);
        index->setParam(parameters);
    }
}