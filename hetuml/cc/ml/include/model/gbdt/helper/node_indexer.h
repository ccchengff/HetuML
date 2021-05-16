#ifndef __HETU_ML_MODEL_GBDT_HELPER_NODE_INDEXER_H_
#define __HETU_ML_MODEL_GBDT_HELPER_NODE_INDEXER_H_

#include "common/logging.h"
#include "common/threading.h"
#include "data/dataset.h"
#include "model/gbdt/model.h"
#include "model/gbdt/hist/grad_pair.h"
#include "model/gbdt/helper/metadata.h"
#include "model/gbdt/helper/tree_sampler.h"
#include <memory>

namespace hetu { 
namespace ml {
namespace gbdt {

/******************************************************
 * Indexing between tree nodes and instances
 ******************************************************/
class NodeIndexer {
public:
  friend class TreeSampler;

  NodeIndexer(int max_depth, uint32_t num_ins) {
    int max_node_num = MAX_NODE_NUM(max_depth);
    node_gps.resize(max_node_num);
    node_start.resize(max_node_num);
    node_end.resize(max_node_num);
    node_actual_end.resize(max_node_num);
    node_to_ins.resize(num_ins);
  }

  void Reset(const InstanceInfo& ins_info) {
    std::fill(node_gps.begin(), node_gps.end(), nullptr);
    std::fill(node_start.begin(), node_start.end(), 0);
    std::fill(node_end.begin(), node_end.end(), 0);
    std::fill(node_actual_end.begin(), node_actual_end.end(), 0);
    node_end[0] = node_actual_end[0] = get_num_ins();
    std::iota(node_to_ins.begin(), node_to_ins.end(), 0);
  }

  const std::shared_ptr<GradPair>& 
  SumNodeGradPairs(nid_t nid, const InstanceInfo& ins_info, 
                   const GBDTParam& param, int class_id = -1) {
    uint32_t from = node_start[nid], until = node_end[nid];
    auto& m1 = ins_info.m1;
    auto& m2 = ins_info.m2;
    int num_classes = param.num_label;
    if (param.is_regression || num_classes == 2) {
      double sum_m1 = 0, sum_m2 = 0;
      #pragma omp parallel for reduction(+:sum_m1,sum_m2)
      for (uint32_t pos_id = from; pos_id < until; pos_id++) {
        uint32_t ins_id = node_to_ins[pos_id];
        sum_m1 += m1[ins_id];
        sum_m2 += m2[ins_id];
      }
      node_gps[nid].reset(new BinaryGradPair(sum_m1, sum_m2));
    } else if (param.multi_tree) {
      // multi-class, using multiple one-vs-rest trees
      ASSERT(class_id >= 0 && class_id < num_classes) 
        << "Error class id: " << class_id;
      uint32_t offset = ins_info.num_ins * class_id;
      double sum_m1 = 0, sum_m2 = 0;
      #pragma omp parallel for reduction(+:sum_m1,sum_m2)
      for (uint32_t pos_id = from; pos_id < until; pos_id++) {
        uint32_t ins_id = node_to_ins[pos_id];
        sum_m1 += m1[offset + ins_id];
        sum_m2 += m2[offset + ins_id];
      }
      node_gps[nid].reset(new BinaryGradPair(sum_m1, sum_m2));
    } else if (ins_info.m1_size == ins_info.m2_size) {
      // multi-class, assuming m2 is diagonal
      std::vector<double> sum_m1(num_classes, 0), sum_m2(num_classes, 0);
      #pragma omp parallel for reduction(vec_double_plus:sum_m1,sum_m2)
      for (uint32_t pos_id = from; pos_id < until; pos_id++) {
        uint32_t ins_id = node_to_ins[pos_id];
        for (int k = 0; k < num_classes; k++) {
          sum_m1[k] += m1[ins_id * num_classes + k];
          sum_m2[k] += m2[ins_id * num_classes + k];
        }
      }
      node_gps[nid].reset(new MultiGradPair(sum_m1, sum_m2));
    } else {
      // multi-class, m1 and m2 have different sizes
      uint32_t m1_size = ins_info.m1_size, m2_size = ins_info.m2_size;
      std::vector<double> sum_m1(m1_size, 0), sum_m2(m2_size, 0);
      #pragma omp parallel for reduction(vec_double_plus:sum_m1,sum_m2)
      for (uint32_t pos_id = from; pos_id < until; pos_id++) {
        uint32_t ins_id = node_to_ins[pos_id];
        for (int k = 0; k < m1_size; k++) 
          sum_m1[k] += m1[ins_id * m1_size + k];
        for (int k = 0; k < m2_size; k++) 
          sum_m2[k] += m2[ins_id * m2_size + k];
      }
      node_gps[nid].reset(new MultiGradPair(sum_m1, sum_m2));
    }
    return get_node_gp_ptr(nid);
  }

  void UpdatePos(nid_t nid, const CompactDataset<float, int>& dataset, 
                 const SplitEntry& split_entry, 
                 const std::vector<float>& splits) {
    int split_fid = split_entry.get_fid();
    
    // get flow to of one instance
    auto GetFlowTo = [&] (uint32_t pos_id) {
      uint32_t ins_id = node_to_ins[pos_id];
      int bin_id = dataset.get(ins_id, split_fid);
      if (bin_id >= 0) 
        return split_entry.FlowTo(splits[bin_id]);
      else 
        return split_entry.DefaultTo();
    };

    // in-place update position
    auto UpdateRange = [&] (uint32_t from, uint32_t until) {
      if (from == until) 
        return from;

      uint32_t left = from, right = until - 1;
      while (left < right) {
        while (left < right && GetFlowTo(left) == 0) left++;
        while (left < right && GetFlowTo(right) == 1) right--;
        if (left < right) {
          std::swap(node_to_ins[left], node_to_ins[right]);
          left++; right--;
        }
      } 
      // find cutting position
      if (left != right) {
        return right;
      } else {
        if (GetFlowTo(left) == 1)
          return left;
        else 
          return left + 1;
      }
    };

    // update sampled part
    uint32_t cut1 = UpdateRange(node_start[nid], node_end[nid]);
    uint32_t num_sampled_l = cut1 - node_start[nid];
    uint32_t num_sampled_r = get_node_size(nid) - num_sampled_l;
    // update unsampled part
    uint32_t cut2 = UpdateRange(node_end[nid], node_actual_end[nid]);
    uint32_t num_unsampled_l = cut2 - node_end[nid];
    uint32_t num_unsampled_r = get_node_actual_size(nid) - 
      get_node_size(nid) - num_unsampled_l;
    // swap right child's sampled part and left child's unsampled part
    for (uint32_t i1 = cut1, i2 = cut2 - 1; 
        i1 < node_end[nid] && i2 >= node_end[nid]; i1++, i2--) {
      std::swap(node_to_ins[i1], node_to_ins[i2]);
    }
    // set edges
    node_start[2 * nid + 1] = node_start[nid];
    node_end[2 * nid + 1] = node_start[2 * nid + 1] + num_sampled_l;
    node_actual_end[2 * nid + 1] = node_end[2 * nid + 1] + num_unsampled_l;
    node_start[2 * nid + 2] = node_actual_end[2 * nid + 1];
    node_end[2 * nid + 2] = node_start[2 * nid + 2] + num_sampled_r;
    node_actual_end[2 * nid + 2] = node_end[2 * nid + 2] + num_unsampled_r;
  }

  void UpdatePreds(nid_t nid, std::vector<float>& preds, 
                   float weight, float learning_rate) const {
    float update = weight * learning_rate;
    uint32_t from = node_start[nid], until = node_actual_end[nid];
    #pragma omp parallel for
    for (uint32_t pos_id = from; pos_id < until; pos_id++) {
      uint32_t ins_id = node_to_ins[pos_id];
      preds[ins_id] += update;
    }
  }

  void UpdatePreds(nid_t nid, std::vector<float>& preds, 
                   const std::vector<float>& weights, 
                   float learning_rate) const {
    int num_classes = weights.size();
    std::vector<float> updates(num_classes);
    for (int k = 0; k < num_classes; k++)
      updates[k] = weights[k] * learning_rate;
    uint32_t from = node_start[nid], until = node_actual_end[nid];
    #pragma omp parallel for
    for (uint32_t pos_id = from; pos_id < until; pos_id++) {
      uint32_t ins_id = node_to_ins[pos_id];
      uint32_t offset = ins_id * num_classes;
      for (int k = 0; k < num_classes; k++)
        preds[offset + k] += updates[k];
    }
  }

  void UpdatePreds(nid_t nid, std::vector<float>& preds, 
                   float weight, float learning_rate, 
                   int num_classes, int class_id) const {
    ASSERT(class_id >= 0 && class_id < num_classes) 
      << "Error class id: " << class_id;
    std::vector<float> updates(num_classes);
    float update = weight * learning_rate;
    uint32_t from = node_start[nid], until = node_actual_end[nid];
    #pragma omp parallel for
    for (uint32_t pos_id = from; pos_id < until; pos_id++) {
      uint32_t ins_id = node_to_ins[pos_id];
      preds[ins_id * num_classes + class_id] += update;
    }
  }

  inline const GradPair& get_node_gp(nid_t nid) const {
    return *node_gps[nid];
  }

  inline void set_node_gp(nid_t nid, const GradPair& gp) {
    this->node_gps[nid].reset(gp.copy());
  }

  inline const std::shared_ptr<GradPair>& get_node_gp_ptr(nid_t nid) const {
    return node_gps[nid];
  }

  inline void set_node_gp_ptr(nid_t nid, std::shared_ptr<GradPair>& gp) {
    this->node_gps[nid] = gp;
  }

  inline uint32_t get_node_start(nid_t nid) const {
    return node_start[nid];
  }

  inline uint32_t get_node_end(nid_t nid) const {
    return node_end[nid];
  }

  inline uint32_t get_node_actual_end(nid_t nid) const {
    return node_actual_end[nid];
  }

  inline uint32_t get_node_size(nid_t nid) const {
    return node_end[nid] - node_start[nid];
  }

  inline uint32_t get_node_actual_size(nid_t nid) const {
    return node_actual_end[nid] - node_start[nid];
  }

  inline const std::vector<uint32_t>& get_node_to_ins() const {
    return node_to_ins;
  }

  inline uint32_t get_num_ins() const { return node_to_ins.size(); }

private:
  std::vector<std::shared_ptr<GradPair>> node_gps;
  std::vector<uint32_t> node_start;
  std::vector<uint32_t> node_end;
  std::vector<uint32_t> node_actual_end;
  std::vector<uint32_t> node_to_ins;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_HELPER_NODE_INDEXER_H_
