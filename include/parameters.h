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

    // 这段代码定义了一个名为 Parameters 的类，用于存储和管理参数集
    class Parameters {
    public:
        // 泛型 set：任何能被 operator<< 写入到 stringstream 的类型
        template<typename T>
        inline void set(const std::string &name, const T &val) {
            std::stringstream ss;
            ss << val;
            params[name] = ss.str();
        }

        // set: std::array<float, N>  ->  "a,b,c,..."
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

        // set: std::vector<float>  ->  "a,b,c,..."
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

        // set: std::vector<std::vector<float>>
        // 约定：行之间用 ';' 分隔，行内元素用 ',' 分隔，例如 "1,2,3;4,5;6,7,8"
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

        // get 泛型：从字符串转为 T（需要 ConvertStrToValue<T> 支持）
        template<typename T>
        inline T get(const std::string &name) const {
            auto item = params.find(name);
            if (item == params.end()) {
                throw std::invalid_argument("Invalid paramter name : " + name + ".");
            } else {
                return ConvertStrToValue<T>(item->second);
            }
        }

        // 把整个参数字典转为 "k:v k2:v2 ..." 的字符串
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

        // 默认的字符串 -> T 转换：依赖 operator>>，适合标量等
        template<typename T>
        inline T ConvertStrToValue(const std::string &str) const {
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

    // 若将来需要 array<float,3>，可参考此注释示例：
    // template<>
    // inline std::array<float,3>
    // Parameters::ConvertStrToValue<std::array<float,3>>(const std::string& str) const {
    //     std::array<float,3> out{0.f, 0.f, 0.f};
    //     std::istringstream is(str);
    //     std::string tok;
    //     size_t i = 0;
    //     while (i < 3 && std::getline(is, tok, ',')) {
    //         out[i] = tok.empty() ? 0.f : std::stof(tok);
    //         ++i;
    //     }
    //     return out;
    // }

    // 专门化：解析 "a,b,c,d" -> std::array<float,4>
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

    // 专门化：解析 "a,b,c,..." -> std::vector<float>
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
                // 与 array 的风格保持一致：空 token 视为 0.f（例如 ",,"）
                out.push_back(0.f);
            }
        }
        return out;
    }

    // 专门化：解析 "a,b; c,d,e; ..." -> std::vector<std::vector<float>>
    // 规则：';' 分隔行，',' 分隔行内元素
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

} // namespace stkq

#endif
