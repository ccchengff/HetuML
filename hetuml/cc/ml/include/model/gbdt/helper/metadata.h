#ifndef __HETU_ML_GBDT_HELPER_METADATA_H
#define __HETU_ML_GBDT_HELPER_METADATA_H

#include "common/logging.h"
#include "data/dataset.h"
#include "stats/quantile.h"
#include <vector>
#include <limits>

namespace hetu { 
namespace ml {
namespace gbdt {

class InstanceInfo {
public:
  InstanceInfo(uint32_t num_ins, uint32_t pred_size)
  : InstanceInfo(num_ins, pred_size, pred_size) {}

  InstanceInfo(uint32_t num_ins, uint32_t m1_size, uint32_t m2_size)
  : num_ins(num_ins), m1_size(m1_size), m2_size(m2_size) {
    this->predictions.resize(num_ins * m1_size);
    this->m1.resize(num_ins * m1_size);
    this->m2.resize(num_ins * m2_size);
    HML_LOG_INFO << "Init InstanceInfo with #ins[" << num_ins << "]";
  }

  const uint32_t num_ins;
  uint32_t m1_size;
  uint32_t m2_size;
  std::vector<float> predictions;
  std::vector<double> m1;
  std::vector<double> m2;
};

class FeatureInfo {
public:
  FeatureInfo(const std::vector<std::vector<float>>& splits, 
              const std::vector<uint32_t>& num_bins, 
              const std::vector<uint32_t>& default_bins)
  : num_feat(splits.size()) {
    ASSERT_GT(num_feat, 0) << "#dimension is zero";
    this->splits.reserve(num_feat);
    this->num_bins.reserve(num_feat);
    this->default_bins.reserve(num_feat);
    this->is_feat_used.reserve(num_feat);
    uint32_t num_total_bins = 0, num_empty = 0;
    for (int fid = 0; fid < MIN(splits.size(), num_feat); fid++) {
      std::vector<float> feat_splits = splits[fid];
      this->splits.push_back(feat_splits);
      this->num_bins.push_back(num_bins[fid]);
      this->default_bins.push_back(default_bins[fid]);
      this->is_feat_used.push_back(num_bins[fid] > 0);
      ASSERT(feat_splits.size() > 0 || num_bins[fid] == 0) 
        << "Feature[" << fid << "] has no splits but " 
        << num_bins[fid] << " bins given";
      if (feat_splits.size() == 0) num_empty++;
      else num_total_bins += num_bins[fid]; 
    }
    HML_LOG_INFO << "Init FeatureInfo with #feat[" << num_feat 
      << "], #bins[" << num_total_bins 
      << "], #empty[" << num_empty << "]";
  }

  const int num_feat;
  std::vector<std::vector<float>> splits;
  std::vector<uint32_t> num_bins;
  std::vector<uint32_t> default_bins;
  std::vector<bool> is_feat_used;
};

class MetaFactory {
public:
  MetaFactory() = delete;

  template <typename V> static FeatureInfo* 
  CreateFeatureInfo(const DataMatrix<V>& data_matrix, int num_splits, 
                    uint32_t max_dim_opt = 0) {
    TIK(create);
    // scan the features to make quantile summaries
    uint32_t num_ins = data_matrix.get_num_instances();
    uint32_t max_dim = max_dim_opt > 0 ? max_dim_opt 
                                       : data_matrix.get_max_dim();
    std::vector<std::shared_ptr<QuantileSketch<V>>> sketches(max_dim);
    const int64_t max_buffer_size = 100000;
    for (uint32_t fid = 0; fid < max_dim; fid++)
      sketches[fid].reset(new HistQuantileSketch<V>(max_buffer_size));
    if (data_matrix.is_dense()) {
      for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
        const auto& dv = data_matrix.get_dense_feature(ins_id);
        for (size_t dim = 0; dim < dv.dim; dim++) {
          if (!std::isnan(dv.values[dim]))
            sketches[dim]->Update(dv.values[dim]);
        }
      }
    } else {
      for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
        const auto& sv = data_matrix.get_sparse_feature(ins_id);
        for (size_t nnz_id = 0; nnz_id < sv.nnz; nnz_id++) {
          if (!std::isnan(sv.values[nnz_id]))
            sketches[sv.indices[nnz_id]]->Update(sv.values[nnz_id]);
        }
      }
    }

    // get candidate splits
    std::vector<std::vector<float>> splits;
    std::vector<uint32_t> num_bins;
    std::vector<uint32_t> default_bins;
    for (uint32_t fid = 0; fid < max_dim; fid++) {
      if (sketches[fid]->empty()) {
        splits.push_back({});
        num_bins.push_back(0);
        default_bins.push_back(0);
      } else {
        std::shared_ptr<QuantileSummary<V>> summary;
        summary.reset(sketches[fid]->MakeSummary());
        std::vector<V> feat_splits;
        if (!summary->TryDistinct(num_splits, feat_splits)) {
          summary->Query(num_splits, feat_splits);  
        }
        std::sort(feat_splits.begin(), feat_splits.end());
        Unique<V>(feat_splits);
        ASSERT_GT(feat_splits.size(), 0) 
          << "Feature[" << fid << "] has no splits";
        std::vector<float> feat_splits_float(feat_splits);
        // make sure there is a bin for zero
        if (feat_splits_float.size() > 1) {
          if (ABS(feat_splits_float.front()) < EPSILON) {
            feat_splits_float.insert(feat_splits_float.begin(), -0.5f);
          } else if (ABS(feat_splits_float.back()) < EPSILON) {
            feat_splits_float.push_back(0.5f);
          }
        } else {
          if (feat_splits_float[0] > EPSILON) {
            feat_splits_float = { -0.5f, feat_splits_float[0] };
          } else if (feat_splits_float[0] < -EPSILON) {
            feat_splits_float = { feat_splits_float[0], 0.5f };
          } else {
            feat_splits_float = { -0.5f, 0.5f };
          }
        }
        splits.push_back(feat_splits_float);
        num_bins.push_back(feat_splits_float.size());
        default_bins.push_back(IndexOf<V>(feat_splits_float, 0));
      }
    }
    auto* ret = new FeatureInfo(splits, num_bins, default_bins);
    TOK(create);
    HML_LOG_INFO << "Create feature info cost " << COST_MSEC(create) << " ms";
    return ret;
  }

  template <typename L, typename V> static CompactDataset<L, int>* 
  CreateBinnedDataset(const Dataset<L, V>& dataset, 
                      const FeatureInfo& feat_info) {
    TIK(binning);
    
    uint32_t num_ins = dataset.get_num_instances();
    uint32_t max_dim = dataset.get_max_dim();
    if (num_ins == 0)
      return new CompactDataset<L, int>();

    // TODO: can we avoid copy?
    std::vector<L> labels(dataset.get_labels());
    std::vector<uint32_t> index_ends;
    std::vector<uint32_t> indices;
    std::vector<int> bins;

    index_ends.resize(num_ins);
    if (dataset.is_dense()) {
      bins.resize(num_ins * max_dim);
      for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
        const auto& dv = dataset.get_dense_feature(ins_id);
        uint32_t offset = ins_id * max_dim;
        for (uint32_t dim = 0; dim < max_dim; dim++) {
          if (std::isnan(dv.values[dim])) {
            bins[offset + dim] = feat_info.default_bins[dim];
          } else {
            bins[offset + dim] = IndexOf<V>(feat_info.splits[dim], 
              dv.values[dim]);
          }
        }
        index_ends[ins_id] = (ins_id + 1) * max_dim;
      }
    } else {
      for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
        const auto& sv = dataset.get_sparse_feature(ins_id);
        indices.insert(indices.end(), sv.indices, sv.indices + sv.nnz);
        for (int nnz_id = 0; nnz_id < sv.nnz; nnz_id++) {
          if (std::isnan(sv.values[nnz_id])) {
            bins.push_back(feat_info.default_bins[sv.indices[nnz_id]]);
          } else {
            bins.push_back(IndexOf<V>(feat_info.splits[sv.indices[nnz_id]], 
              sv.values[nnz_id]));
          }
        }
        index_ends[ins_id] = indices.size();
      }
    }
    
    auto* ret = new CompactDataset<L, int>(labels, index_ends, indices, bins);
    TOK(binning);
    HML_LOG_INFO << "Binning dataset cost " << COST_MSEC(binning) << " ms";
    return ret;
  }
};


} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_GBDT_HELPER_METADATA_H
