#ifndef __HETU_ML_MODEL_GBDT_TRAIN_TRAINER_H_
#define __HETU_ML_MODEL_GBDT_TRAIN_TRAINER_H_

#include "common/logging.h"
#include "common/threading.h"
#include "data/dataset.h"
#include "model/common/mlbase.h"
#include "model/gbdt/model.h"
#include "model/gbdt/helper/tree_booster.h"
#include "model/gbdt/helper/tree_sampler.h"
#include "model/gbdt/helper/node_indexer.h"
#include "model/gbdt/helper/hist_manager.h"
#include "model/gbdt/helper/split_finder.h"
#include <map>
#include <memory>

namespace hetu { 
namespace ml {
namespace gbdt {

typedef std::shared_ptr<CompactDataset<label_t, int>> TrainSetPtr;
typedef const Dataset<label_t, float> ValidSet;

typedef std::shared_ptr<GradPair> GradPairPtr;
typedef std::shared_ptr<GBTSplit> GBTSplitPtr;
typedef std::unique_ptr<GBTNode> GBTNodePtr;
typedef std::unique_ptr<GBTree> GBTreePtr;

typedef std::vector<GBTNodePtr> GBTNodePtrVec;
typedef std::vector<GBTSplitPtr> GBTSplitPtrVec;
typedef std::shared_ptr<GBTSplitPtrVec> GBTSplitPtrVecPtr;
typedef std::map<nid_t, GBTSplitPtr> GBTSplitPtrMap;
typedef std::shared_ptr<GBTSplitPtrMap> GBTSplitPtrMapPtr;

class GBDTTrainer {
public:
  GBDTTrainer(const GBDTParam& param) {
    this->param.reset(new GBDTParam(param));
  }

  ~GBDTTrainer() {
    if (this->train_data != nullptr)
      delete this->train_data;
  }

  GBDTModel* Fit(const Dataset<label_t, float>& train_data, 
                 const Dataset<label_t, float>& valid_data) {
    this->Init(train_data, valid_data);
    return this->FitModel();
  }

protected:

  virtual FeatureInfo*
  GetFeatureInfo(const Dataset<label_t, float>& dataset) {
    return MetaFactory::CreateFeatureInfo(dataset, this->param->num_split);
  }

  virtual void Init(const Dataset<label_t, float>& train_data, 
                    const Dataset<label_t, float>& valid_data) {
    // meta data
    this->ins_info.reset(new InstanceInfo(train_data.get_num_instances(), 
      param->pred_size(), param->m2_size()));
    this->feat_info.reset(GetFeatureInfo(train_data));
    // train set
    this->train_data = MetaFactory::CreateBinnedDataset(
      train_data, *this->feat_info);
    // TODO: binned train data
    this->num_train = this->train_data->get_num_instances();
    // valid set
    this->valid_data = &valid_data;
    this->num_valid = this->valid_data->get_num_instances();
    this->valid_preds.resize(this->num_valid * this->param->pred_size());
    // init helpers
    this->tree_booster.reset(new TreeBooster(this->param));
    this->tree_sampler.reset(new TreeSampler(
      this->param->ins_sp_ratio, this->param->feat_sp_ratio));
    this->node_indexer.reset(new NodeIndexer(
      this->param->max_depth, this->num_train));
    this->hist_manager.reset(new HistManager(this->param->max_depth, 
      this->param->num_label, this->param->multi_tree));
    this->split_finder.reset(new SplitFinder(
      this->param->num_label, this->param->multi_tree, 
      this->param->min_split_gain, this->param->reg_lambda));
  }

  /* -------------------- Training Routines -------------------- */
  GBDTModel* FitModel();

  void InitPreds(GBDTModel& model);

  void NewRound(int round_id);

  void NewTree(GBTNodePtrVec& node_buffer, int class_id = -1);

  GBTSplitPtr FindRootSplit(const GBTNode& root, int class_id = -1);

  void FindSplits(const std::vector<nid_t>& to_find, 
                  GBTNodePtrVec& node_buffer, 
                  GBTSplitPtrVec& split_buffer, 
                  int class_id = -1);

  void ChooseSplits(std::vector<nid_t>& to_find, 
                    std::vector<nid_t>& to_split, 
                    std::vector<nid_t>& to_set_leaves, 
                    GBTSplitPtrVec& split_buffer, 
                    int cur_node_num);

  void SplitNodes(GBTree& tree, const std::vector<nid_t>& to_split, 
                  GBTNodePtrVec& node_buffer, GBTSplitPtrVec& split_buffer, 
                  int class_id = -1);

  void SetAsLeaves(GBTree& tree, const std::vector<nid_t>& to_set_leaves, 
                   GBTNodePtrVec& node_buffer);

  void FinishTree(GBTree& tree, GBTNodePtrVec& node_buffer);

  void UpdatePreds(GBTree& tree, float learning_rate, int class_id = -1);

  void Evaluate(int round_id);

  inline bool NodeCanSplit(const GBTNode& node) {
    return node.get_size() > param->min_node_ins 
      && tree_booster->SatisfyWeight(node.get_sum_gp());
  }

  std::shared_ptr<GBDTParam> param;
  
  CompactDataset<label_t, int>* train_data;
  const Dataset<label_t, float>* valid_data;
  std::vector<float> valid_preds;
  size_t num_train;
  size_t num_valid;

  std::shared_ptr<InstanceInfo> ins_info;
  std::shared_ptr<FeatureInfo> feat_info;

  std::shared_ptr<TreeBooster> tree_booster;
  std::shared_ptr<TreeSampler> tree_sampler;
  std::shared_ptr<NodeIndexer> node_indexer;
  std::shared_ptr<HistManager> hist_manager;
  std::shared_ptr<SplitFinder> split_finder;

  /* -------------------- Training APIs -------------------- */

  virtual void DoInitPreds(std::vector<float>& init_preds);
  
  virtual void DoNewRound();

  virtual std::tuple<uint32_t, GradPairPtr> 
  DoNewTree(int class_id = -1, uint64_t ins_sp_seed = 0, 
            uint64_t feat_sp_seed = 0);

  virtual GBTSplitPtr 
  DoFindRootSplit(const GradPair& root_gp, const float root_gain, 
                  int class_id = -1);
  
  virtual GBTSplitPtrMapPtr 
  DoFindSplits(const std::vector<nid_t>& nids, 
               const std::vector<bool>& can_splits, 
               const std::vector<nid_t>& to_build, 
               const std::vector<nid_t>& to_subtract, 
               const std::vector<nid_t>& to_remove, 
               const std::vector<GradPairPtr>& node_sum_gps, 
               const std::vector<float>& node_gains, 
               int class_id = -1);
    
  virtual std::map<uint32_t, uint32_t> 
  DoSplitNodes(const GBTSplitPtrMap& splits, int class_id = -1);

  virtual void DoSetAsLeaves(const std::vector<nid_t>& nids);

  virtual void DoUpdatePreds(GBTree& tree, float lr, int class_id = -1);

  virtual std::map<std::string, std::tuple<double, double>> 
  DoEvaluate(int round_id);
};

static inline void CheckLabels(const std::vector<label_t>& labels, 
                               int num_classes) {
  for (auto label : labels) {
    int label_int = static_cast<int>(label);
    ASSERT(label_int >= 0 && label_int < num_classes) 
      << "Invalid label: " << label_int 
      << ", should be in [0, " << num_classes - 1 << "]";
  }
}

static inline void CheckAndCountLabels(const std::vector<label_t>& labels, 
                                       std::vector<uint32_t>& cnts, 
                                       int num_classes) {
  cnts.resize(num_classes);
  for (auto label : labels) {
    int label_int = static_cast<int>(label);
    ASSERT(label_int >= 0 && label_int < num_classes) 
      << "Invalid label: " << label_int 
      << ", should be in [0, " << num_classes - 1 << "]";
    cnts[label_int]++;
  }
}

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_TRAIN_TRAINER_H_
