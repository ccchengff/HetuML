#ifndef __HETU_ML_MODEL_COMMON_ARGPARSE_H_
#define __HETU_ML_MODEL_COMMON_ARGPARSE_H_

#include "common/logging.h"
#include <vector>
#include <unordered_map>
#include <iomanip>

namespace hetu { 
namespace ml {

using Args = std::unordered_map<std::string, std::string>;

namespace argparse {

inline static bool ParseBool(const std::string& str) {
  bool result = false;
  std::istringstream is(str);
  is >> result; // "0" or "1"
  if (is.fail()) {
    is.clear();
    is >> std::boolalpha >> result; // "true" or "false"
  }
  ASSERT(!is.fail()) << "Cannot convert string '" << str << "' to bool";
  return result;
}

template <typename T>
inline T Parse(const std::string& str) {
  T result;
  std::istringstream is(str);
  is >> result;
  ASSERT(!is.fail()) << "Cannot convert '" << str << "'";
  return result;
}

template <typename T>
inline void ParseVector(const std::string& str, std::vector<T>& vec) {
  ASSERT(str[0] == '[' && str[str.length() - 1] == ']')
    << "Error parsing '" << str << "' as std::vector";
  std::string vec_str = str.substr(1, str.length() - 2);
  std::string temp;
  vec.clear();
  for(std::stringstream ss(vec_str); getline(ss, temp, ',');) 
    vec.push_back(Parse<T>(temp));
}

inline bool GetBool(const Args& args, const std::string& key) {
  auto iter = args.find(key);
  ASSERT(iter != args.end()) << "Key '" << key << "' does not exist";
  return ParseBool(iter->second);
}

inline bool GetBool(const Args& args, const std::string& key, 
                    bool default_value) {
  auto iter = args.find(key);
  if (iter != args.end())
    return ParseBool(iter->second);
  else
    return default_value;
}

template <typename T>
inline T Get(const Args& args, const std::string& key) {
  auto iter = args.find(key);
  ASSERT(iter != args.end()) << "Key '" << key << "' does not exist";
  return Parse<T>(iter->second);
}

template <typename T>
inline T Get(const Args& args, const std::string& key, T default_value) {
  auto iter = args.find(key);
  if (iter != args.end())
    return Parse<T>(iter->second);
  else
    return default_value;
}

inline std::vector<bool> GetVector(const Args& args, const std::string& key, 
                                   char delim = ',') {
  std::vector<bool> res;
  auto iter = args.find(key);
  if (iter != args.end()) {
    const auto& str = iter->second;
    std::string temp;
    for (std::stringstream ss(str); getline(ss, temp, delim);) {
      res.push_back(ParseBool(temp));
    }
  }
  return std::move(res);
}

template <typename T>
inline std::vector<T> GetVector(const Args& args, const std::string& key, 
                                char delim = ',') {
  std::vector<T> res;
  auto iter = args.find(key);
  if (iter != args.end()) {
    const auto& str = iter->second;
    std::string temp;
    for (std::stringstream ss(str); getline(ss, temp, delim);) {
      res.push_back(Parse<T>(temp));
    }
  }
  return std::move(res);
}

inline size_t GetOffset(const std::string& str, const char* prefix, 
                        size_t offset = 0) {
  size_t t = str.find(prefix, offset);
  ASSERT_GE(t, 0) << "Cannot find prefix '" << prefix 
    << "' in string '" << str << "' with offset " << offset;
  return t + strlen(prefix);
}

} // namespace argparse
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_COMMON_ARGPARSE_H_
