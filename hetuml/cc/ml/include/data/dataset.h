#ifndef __HETU_ML_DATA_DATASET_H_
#define __HETU_ML_DATA_DATASET_H_

#include "common/logging.h"
#include "common/math.h"
#include "data/vectors.h"
#include "data/io.h"

namespace hetu { 
namespace ml {

template <typename L> class LabelSet;
template <typename V> class DataMatrix;
template <typename V> class CompactDataMatrix;
template <typename L, typename V> class Dataset;
template <typename L, typename V> class CompactDataset;

template <typename L>
class LabelSet {
public:
  LabelSet() {}

  LabelSet(std::vector<L>& labels): labels{ std::move(labels) } {
    ASSERT(!this->labels.empty()) << "LabelSet has no instances";
  }

  inline const std::vector<L>& get_labels() const { return labels; }

  inline L get_label(uint32_t ins) const { return labels[ins]; }

  inline uint32_t get_num_instances() const { return labels.size(); }

protected:
  std::vector<L> labels;
};

template <typename V>
class DataMatrix {
public:
  DataMatrix(): max_dim(-1), dense(false) {}

  DataMatrix(std::vector<AVector<V>*>& features)
  : features{ std::move(features) } {
    ASSERT(!this->features.empty()) << "DataMatrix has no instances";
    this->max_dim = this->features[0]->dim;
    this->dense = this->features[0]->is_dense();
    for (size_t i = 1; i < this->features.size(); i++) {
      ASSERT(this->dense == this->features[i]->is_dense()) 
        << "Provided dense and sparse vectors simultaneously";
      if (this->dense) {
        ASSERT(this->max_dim == this->features[i]->dim)
          << "Provided dense features with different dimensions: "
          << this->max_dim << " vs. " << this->features[i]->dim;
      } else {
        this->max_dim = MAX(this->max_dim, this->features[i]->dim);
      }
    }
  }

  ~DataMatrix() {
    for (AVector<V>* feature : this->features)
      delete feature;
  }

  inline const std::vector<AVector<V>*>& get_features() const { 
    return features; 
  }

  inline const DenseVector<V>& get_dense_feature(uint32_t ins) const { 
    return static_cast<const DenseVector<V>&>(*features[ins]);
  }

  inline const SparseVector<V>& get_sparse_feature(uint32_t ins) const { 
    return static_cast<const SparseVector<V>&>(*features[ins]);
  }

  inline uint32_t get_num_instances() const { return features.size(); }

  inline uint32_t get_max_dim() const { return max_dim; }

  inline bool is_dense() const { return this->dense; }

  // static DataMatrix<V>* 
  // LoadData(const std::string& path, const std::string& data_format, 
  //          size_t rank = 0, size_t num_workers = 1);

protected:
  std::vector<AVector<V>*> features;
  uint32_t max_dim;
  bool dense;
};

template <typename V>
class CompactDataMatrix {
public:
  CompactDataMatrix(): max_dim(-1) {}

  CompactDataMatrix(std::vector<uint32_t>& index_ends, 
                    std::vector<uint32_t>& indices, 
                    std::vector<V>& values)
  : index_ends{ std::move(index_ends) }, indices{ std::move(indices) }, 
  values{ std::move(values) } {
    ASSERT(!this->index_ends.empty()) << "DataMatrix has no instances";
    ASSERT(!this->values.empty()) << "DataMatrix sparsity is 100%";
    if (this->indices.empty()) {  // dense format
      ASSERT(this->values.size() % this->index_ends.size() == 0) 
        << "#instances and #values not matching for dense dataset: "
        << this->index_ends.size() << " vs. " << this->values.size();
      this->max_dim = this->values.size() / this->index_ends.size();
    } else {  // sparse format
      ASSERT(this->indices.size() == this->values.size()) 
        << "#indices and #values not matching for sparse dataset: " 
        << this->indices.size() << " vs. " << this->values.size();
      this->max_dim = *std::max_element(this->indices.begin(), 
        this->indices.end()) + 1;
    }
  }

  ~CompactDataMatrix() {}

  inline V get(uint32_t ins, uint32_t fid) const {
    if (is_dense()) {
      return values[ins * max_dim + fid];
    } else {
      uint32_t start = (ins == 0) ? 0 : index_ends[ins - 1];
      uint32_t end = index_ends[ins];
      auto begin = indices.begin();
      auto from = begin + start, last = begin + end;
      auto it = std::lower_bound(from, last, fid);
      // TODO: provide default value by argument
      return (it == last || *it != fid) ? ((V) -1) : values[it - begin];
    }
  }

  inline const std::vector<uint32_t>& get_index_ends() const {
    return index_ends;
  }

  inline uint32_t get_index_end(uint32_t ins) const { return index_ends[ins]; }

  inline const std::vector<uint32_t>& get_indices() const { return indices; }

  inline uint32_t get_indice(uint32_t index) const { return indices[index]; }

  inline const std::vector<V>& get_values() const { return values; }

  inline V get_value(uint32_t index) const { return values[index]; }

  inline uint32_t get_num_instances() const { return index_ends.size(); }

  inline uint32_t get_max_dim() const { return max_dim; }

  inline uint32_t get_num_kv() const { return values.size(); }

  inline double get_density() const { 
    return 1.0 * get_num_kv() / (get_num_instances() * get_max_dim());
  }

  inline bool is_dense() const { return indices.empty(); }

protected:
  std::vector<uint32_t> index_ends;
  std::vector<uint32_t> indices;
  std::vector<V> values;
  uint32_t max_dim;
};

template <typename L, typename V>
class Dataset : public LabelSet<L>, public DataMatrix<V> {
public:
  Dataset(): LabelSet<L>(), DataMatrix<V>() {}
  
  Dataset(std::vector<L>& labels, std::vector<AVector<V>*>& features)
  : LabelSet<L>(labels), DataMatrix<V>(features) {
    ASSERT(this->labels.size() == this->features.size()) 
      << "#instances and #features not matching: " 
      << this->labels.size() << " vs. " << this->features.size();
  }

  using LabelSet<L>::get_num_instances;

  static Dataset<L, V>* 
  LoadData(const std::string& path, const std::string& data_format, 
           bool neg_y = false, size_t rank = 0, size_t num_workers = 1);
};

template <typename L, typename V>
class CompactDataset : public LabelSet<L>, public CompactDataMatrix<V> {
public:
  CompactDataset(): LabelSet<L>(), CompactDataMatrix<V>() {}

  CompactDataset(std::vector<L>& labels, std::vector<uint32_t>& index_ends, 
                 std::vector<uint32_t>& indices, std::vector<V>& values)
  : LabelSet<L>(labels), CompactDataMatrix<V>(index_ends, indices, values) {
    ASSERT(this->labels.size() == this->index_ends.size()) 
      << "#instances and #features not matching: " 
      << this->labels.size() << " vs. " << this->index_ends.size();
  }

  using LabelSet<L>::get_num_instances;

  static CompactDataset<L, V>*
  LoadData(const std::string& path, const std::string& data_format, 
           bool neg_y = false, size_t rank = 0, size_t num_workers = 1);
  
  static CompactDataset<L, V>*
  FromDataset(const Dataset<L, V>& dataset);
};

template <typename L, typename V> Dataset<L, V>* 
Dataset<L, V>::LoadData(const std::string& path, 
                        const std::string& data_format, bool neg_y, 
                        size_t rank, size_t num_workers) {
  std::vector<L> labels;
  std::vector<AVector<V>*> features;
  if (!data_format.compare("csv")) {
    V missing_value = std::numeric_limits<V>::quiet_NaN();
    ReadCSVData(labels, features, path, missing_value, neg_y, 
      rank, num_workers);
  } else if (!data_format.compare("libsvm")) {
    ReadLibsvmData(labels, features, path, neg_y, rank, num_workers);
  } else {
    ASSERT(false) << "Unsupported data format: " << data_format;
  }
  return new Dataset<L, V>(labels, features);
}

template <typename L, typename V> CompactDataset<L, V>*
CompactDataset<L, V>::LoadData(const std::string& path, 
                               const std::string& data_format, bool neg_y, 
                               size_t rank, size_t num_workers) {
  auto* dataset = Dataset<L, V>::LoadData(path, data_format, neg_y, 
    rank, num_workers);
  auto* compact = CompactDataset<L, V>::FromDataset(*dataset);
  delete dataset;
  return compact;
}

template <typename L, typename V> CompactDataset<L, V>* 
CompactDataset<L, V>::FromDataset(const Dataset<L, V>& dataset) {
  TIK(compact);
  
  size_t num_ins = dataset.get_num_instances();
  size_t max_dim = dataset.get_max_dim();
  if (num_ins == 0)
    return new CompactDataset<L, V>();

  // TODO: can we avoid copy?
  std::vector<L> labels(dataset.get_labels());
  std::vector<uint32_t> index_ends;
  std::vector<uint32_t> indices;
  std::vector<V> values;

  index_ends.resize(num_ins);
  if (dataset.is_dense()) {
    values.resize(num_ins * max_dim);
    for (size_t ins_id; ins_id < num_ins; ins_id++) {
      const auto& dv = dataset.get_dense_feature(ins_id);
      std::copy(dv.values, dv.values + max_dim, 
        values.begin() + ins_id * max_dim);
      index_ends[ins_id] = (ins_id + 1) * max_dim;
    }
  } else {
    for (size_t ins_id; ins_id < num_ins; ins_id++) {
      const auto sv = dataset.get_sparse_feature(ins_id);
      indices.insert(indices.end(), sv.indices, sv.indices + sv.nnz);
      values.insert(values.end(), sv.values, sv.values + sv.nnz);
      index_ends[ins_id] = indices.size();
    }
  }

  auto* ret = new CompactDataset<L, V>(labels, index_ends, indices, values);
  TOK(compact);
  HML_LOG_INFO << "Compact dataset cost " << COST_MSEC(compact) << " ms";
  return ret;
}

} // namespace ml
} // namespace hetu

# endif // __HETU_ML_DATA_DATASET_H_
