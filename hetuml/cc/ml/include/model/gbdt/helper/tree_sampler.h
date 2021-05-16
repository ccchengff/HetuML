#ifndef __HETU_ML_MODEL_GBDT_HELPER_TREE_SAMPLER_H_
#define __HETU_ML_MODEL_GBDT_HELPER_TREE_SAMPLER_H_

#include "common/logging.h"
#include "common/threading.h"
#include "model/gbdt/model.h"
#include "model/gbdt/helper/metadata.h"
#include "model/gbdt/helper/node_indexer.h"

namespace hetu { 
namespace ml {
namespace gbdt {

/******************************************************
 * Instance & feature sampling
 ******************************************************/
class TreeSampler {
public:

  TreeSampler(float ins_ratio, float feat_ratio)
  : ins_ratio(ins_ratio), feat_ratio(feat_ratio) {
    ASSERT(ins_ratio > 0 && ins_ratio <= 1.0) 
      << "Invalid instance sampling ratio: " << ins_ratio;
    ASSERT(feat_ratio > 0 && feat_ratio <= 1.0) 
      << "Invalid instance sampling ratio: " << feat_ratio;
  }

  bool InsSampling(InstanceInfo& ins_info, NodeIndexer& node_indexer, 
      int class_id = -1, uint64_t seed_opt = 0) {
    if (ShouldSampleIns(node_indexer.get_num_ins())) 
      return DoInsSampling(ins_info, node_indexer, class_id, seed_opt);
    else 
      return false;
  }

  bool FeatSampling(FeatureInfo& feat_info, uint64_t seed_opt) {
    if (ShouldSampleFeat(feat_info.num_feat))
      return DoFeatSampling(feat_info, seed_opt);
    else 
      return false;
  }

protected:

  inline bool ShouldSampleIns(uint32_t num_ins) { 
    return ins_ratio < 1 && ceil(ins_ratio * num_ins) < num_ins;
  }

  inline bool ShouldSampleFeat(int num_feat) {
    return feat_ratio < 1 && ceil(feat_ratio * num_feat) < num_feat;
  }

  virtual bool 
  DoInsSampling(InstanceInfo& ins_info, NodeIndexer& node_indexer, 
                int class_id = -1, uint64_t seed_opt = 0) {
    uint32_t num_ins = ins_info.num_ins;
    uint32_t num_sampled = ceil(ins_ratio * num_ins);
    auto& node_to_ins = node_indexer.node_to_ins;
    // random shuffle
    std::default_random_engine engine;
    if (seed_opt != 0)
      engine.seed(seed_opt);
    else
      engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(node_to_ins.begin(), node_to_ins.end(), engine);
    // sort
    std::sort(node_to_ins.begin(), node_to_ins.begin() + num_sampled);
    std::sort(node_to_ins.begin() + num_sampled, node_to_ins.end());
    // set node end
    node_indexer.node_end[0] = num_sampled;
    return true;
  }
    
  virtual bool DoFeatSampling(FeatureInfo& feat_info, uint64_t seed_opt = 0) {
    int num_feat = feat_info.num_feat;
    auto& num_bins = feat_info.num_bins;
    auto& is_feat_used = feat_info.is_feat_used;
    std::default_random_engine engine;
    if (seed_opt != 0)
      engine.seed(seed_opt);
    else
      engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::bernoulli_distribution bernoulli(feat_ratio);
    for (int fid = 0; fid < num_feat; fid++) {
      is_feat_used[fid] = (num_bins[fid] > 0) && bernoulli(engine);
    }
    return true;
  }

  float ins_ratio;
  float feat_ratio;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_HELPER_TREE_SAMPLER_H_
