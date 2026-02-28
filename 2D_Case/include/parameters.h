#ifndef STKQ_PARAMETERS_H
#define STKQ_PARAMETERS_H

#include <sstream>
#include <unordered_map>
#include <array>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <typeinfo>
#include <vector>


namespace stkq {
    class Parameters {
    public:
        template<typename T>  
        inline void set(const std::string &name, const T &val) { 
            std::stringstream ss;  
            ss << val;  
            params[name] = ss.str();
        }

        template<size_t N>
        inline void set(const std::string& name,
                                    const std::array<float,N>& a) {
            std::ostringstream ss;
            ss << std::setprecision(9) << std::defaultfloat;
            for (size_t i = 0; i < N; ++i) {
                if (i) ss << ',';
                ss << a[i];
            }
            params[name] = ss.str();
        }

        
        inline void set(const std::string& name,
                        const std::vector<float>& vec) {
            std::ostringstream ss;
            ss << std::setprecision(9) << std::defaultfloat;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (i) ss << ',';
                ss << vec[i];
            }
            params[name] = ss.str();
        }

        inline void set(const std::string& name,
                        const std::vector<std::vector<float>>& vvec) {
            std::ostringstream ss;
            ss << std::setprecision(9) << std::defaultfloat;
            for (size_t r = 0; r < vvec.size(); ++r) {
                if (r) ss << ';';
                const auto& row = vvec[r];
                for (size_t c = 0; c < row.size(); ++c) {
                    if (c) ss << ',';
                    ss << row[c];
                }
            }
            params[name] = ss.str();
        }

        template<typename T>
        inline T get(const std::string &name) const { 
            auto item = params.find(name);
            if (item == params.end()) {
                throw std::invalid_argument("Invalid paramter name : " + name + ".");
            } else {
                return ConvertStrToValue<T>(item->second); 
            }
        }

        inline std::string toString() const {  
            std::string res;
            for (auto &param : params) { 
                res += param.first;
                res += ":";
                res += param.second;
                res += " ";
            }
            return res;
        }

    private:
        std::unordered_map<std::string, std::string> params;

        template<typename T>
        inline T ConvertStrToValue(const std::string &str) const { // 字符串-> 指定类型值
            std::stringstream sstream(str);
            T value;
            if (!(sstream >> value) || !sstream.eof()) {
                std::stringstream err;
                err << "Fail to convert value" << str << " to type: " << typeid(value).name();
                throw std::runtime_error(err.str());
            }
            return value;
        }
    };



    template<>
    inline std::array<float,4>
    Parameters::ConvertStrToValue<std::array<float,4>>(const std::string& str) const {
        std::array<float,4> out{0.f, 0.f, 0.f, 0.f};
        std::istringstream is(str);
        std::string tok;
        size_t i = 0;
        while (i < 4 && std::getline(is, tok, ',')) {
            out[i] = tok.empty() ? 0.f : std::stof(tok);
            ++i;
        }
        return out;
    }


    template<>
    inline std::vector<float>
    Parameters::ConvertStrToValue<std::vector<float>>(const std::string& str) const {
        std::vector<float> out;
        std::istringstream is(str);
        std::string tok;
        while (std::getline(is, tok, ',')) {
            if (!tok.empty()) {
                out.push_back(std::stof(tok));
            } else {
                out.push_back(0.f);
            }
        }
        return out;
    }

    template<>
    inline std::vector<std::vector<float>>
    Parameters::ConvertStrToValue<std::vector<std::vector<float>>>(const std::string& str) const {
        std::vector<std::vector<float>> out;
        std::istringstream is_rows(str);
        std::string row_str;
        while (std::getline(is_rows, row_str, ';')) {
            std::vector<float> row;
            std::istringstream is_vals(row_str);
            std::string tok;
            while (std::getline(is_vals, tok, ',')) {
                if (!tok.empty()) {
                    row.push_back(std::stof(tok));
                } else {
                    row.push_back(0.f);
                }
            }
            out.emplace_back(std::move(row));
        }
        return out;
    }
    
}

#endif

