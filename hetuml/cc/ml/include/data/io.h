#ifndef __HETU_ML_MODEL_DATA_IO_H_
#define __HETU_ML_MODEL_DATA_IO_H_

#include "common/logging.h"
#include "data/vectors.h"
#include <memory>
#include <fstream>
#include <vector>

namespace hetu { 
namespace ml {

inline static void split(const std::string& s, char delim, bool drop_empty, 
                         std::vector<std::string>& elems) {
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)) {
    if (item.length() > 0 || !drop_empty)
      elems.push_back(item);
  }
}

inline static void split(const std::string& s, char delim,
                         std::vector<std::string>& elems) {
  split(s, delim, false, elems);
};

template <typename L, typename V>
static void ReadCSVData(std::vector<L>& labels, 
                        std::vector<AVector<V>*>& features, 
                        const std::string& path, 
                        const V missing, bool neg_y = false, 
                        size_t rank = 0, size_t num_workers = 1) {
  TIK(load);
  std::ifstream fin(path, std::ifstream::in);
  ASSERT(fin) << "Failed to open " << path;
  
  std::string line;
  std::vector<std::string> tokens;
  size_t line_num = 0;
  while (getline(fin, line)) {
    if (line.length() > 0 && line[0] != '#' && line[0] != ' ') {
      line_num++;
      if (num_workers > 1 && (line_num - 1) % num_workers != rank) 
        continue;
      tokens.clear();
      split(line, ',', false, tokens);
      L label = atof(tokens[0].c_str());
      if (neg_y && label != 1) {
        label = -1;
      }
      size_t num_features = tokens.size() - 1;
      V *values = new V[num_features];
      for (size_t i = 0; i < num_features; i++) {
        if (tokens[i + 1].length() == 0) {
          values[i] = missing;
        } else {
          values[i] = atof(tokens[i + 1].c_str());
        }
      }
      size_t dim = num_features;
      auto* dv = new DenseVector<V>(values, dim);
      labels.push_back(label);
      features.push_back(dv);
    }
  }
  
  fin.close();
  TOK(load);
  HML_LOG_INFO << "Read csv data from " << path 
    << " cost " << COST_MSEC(load) << " ms";
}

// read libsvm data (sparse format)
template <typename L, typename V>
static void ReadLibsvmData(std::vector<L>& labels, 
                           std::vector<AVector<V>*>& features, 
                           const std::string& path, bool neg_y = false, 
                           size_t rank = 0, size_t num_workers = 1) {
  TIK(load);
  std::ifstream fin(path, std::ifstream::in);
  ASSERT(fin) << "Failed to open " << path;

  std::string line;
  std::vector<std::string> tokens;
  std::vector<std::string> feat_val(2);
  size_t line_num = 0;
  while (getline(fin, line)) {
    if (line.length() > 0 && line[0] != '#' && line[0] != ' ') {
      line_num++;
      if (num_workers > 1 && (line_num - 1) % num_workers != rank)
        continue;
      tokens.clear();
      split(line, ' ', true, tokens);
      L label = atof(tokens[0].c_str());
      if (neg_y && label != 1) {
        label = -1;
      }
      int nnz = tokens.size() - 1;
      int *indices = new int[nnz];
      V *values = new V[nnz];
      for (size_t i = 0; i < nnz; i++) {
        feat_val.clear();
        split(tokens[i + 1], ':', true, feat_val);
        ASSERT(feat_val.size() == 2) << "Cannot parse " << tokens[i + 1];
        indices[i] = atoi(feat_val[0].c_str());
        values[i] = atof(feat_val[1].c_str());
      }
      int dim = nnz > 0 ? (indices[nnz - 1] + 1) : 1;
      auto* sv = new SparseVector<V>(indices, values, nnz, dim);
      labels.push_back(label);
      features.push_back(sv);
    }
  }
  fin.close();
  TOK(load);
  HML_LOG_INFO << "Read libsvm data from " << path 
    << " cost " << COST_MSEC(load) << " ms";
};

} // namespace ml
} // namespace hetu

# endif // __HETU_ML_MODEL_DATA_IO_H_
