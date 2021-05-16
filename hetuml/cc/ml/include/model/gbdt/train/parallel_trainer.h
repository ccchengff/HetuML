#ifndef __HETU_ML_MODEL_GBDT_TRAIN_PARALLEL_TRAINER_H_
#define __HETU_ML_MODEL_GBDT_TRAIN_PARALLEL_TRAINER_H_

#include "model/gbdt/train/trainer.h"
#include "ps/psmodel/PSMatrix.h"
#include "ps/psmodel/PSVector.h"

namespace hetu { 
namespace ml {
namespace gbdt {

#define PsDataType double
typedef std::shared_ptr<PSVector<PsDataType>> PsVectorPtr;
typedef std::shared_ptr<PSMatrix<PsDataType>> PsMatrixPtr;

class GBDTDPTrainer : public GBDTTrainer {
public:
  GBDTDPTrainer(const GBDTParam& param): GBDTTrainer(param) {
    this->rank = MyRank();
    this->num_workers = NumWorkers();
  }

protected:
  FeatureInfo*
  GetFeatureInfo(const Dataset<label_t, float>& dataset) override {
    // sync max dim in order to avoid data skewness
    int rank = MyRank();
    int num_workers = NumWorkers();
    std::vector<PsDataType> max_dims(num_workers);
    PSVector<PsDataType> ps_max_dims("max_dims", num_workers);
    if (rank == 0) ps_max_dims.initAllZeros();
    GlobalSync();
    auto local_max_dim = static_cast<PsDataType>(dataset.get_max_dim());
    ps_max_dims.sparsePush(&rank, &local_max_dim, 1, false);
    GlobalSync();
    ps_max_dims.densePull(max_dims.data(), num_workers);
    auto global_max_dim = static_cast<size_t>(*std::max_element(
      max_dims.begin(), max_dims.end()));
    
    // init ps vectors to store candidate splits
    PsVectorPtr ps_splits(new PSVector<PsDataType>(
      "candidate_splits", global_max_dim * (param->num_split + 1)));
    PsVectorPtr ps_num_bins(new PSVector<PsDataType>(
      "num_bins", global_max_dim));
    PsVectorPtr ps_default_bins(new PSVector<PsDataType>(
      "default_bins", global_max_dim));
    
    FeatureInfo* feat_info = nullptr;
    if (this->rank == 0) {
      // propose candidate splits
      feat_info = MetaFactory::CreateFeatureInfo(
        dataset, this->param->num_split, global_max_dim);
      // push to PS
      std::vector<PsDataType> all_splits(global_max_dim * 
        (param->num_split + 1));
      for (size_t fid = 0, offset = 0; fid < global_max_dim; fid++) {
        std::copy(feat_info->splits[fid].begin(), feat_info->splits[fid].end(), 
          all_splits.begin() + offset);
        offset += feat_info->num_bins[fid];
      }
      std::vector<PsDataType> num_bins(feat_info->num_bins.begin(), 
        feat_info->num_bins.end());
      std::vector<PsDataType> default_bins(feat_info->default_bins.begin(), 
        feat_info->default_bins.end());
      ps_splits->initAllZeros();
      ps_num_bins->initAllZeros();
      ps_default_bins->initAllZeros();
      ps_splits->densePush(all_splits.data(), all_splits.size());
      ps_num_bins->densePush(num_bins.data(), num_bins.size());
      ps_default_bins->densePush(default_bins.data(), default_bins.size());
      GlobalSync();
    } else {
      // pull from PS
      GlobalSync();
      std::vector<PsDataType> all_splits(global_max_dim * 
        (param->num_split + 1));
      std::vector<PsDataType> num_bins(global_max_dim);
      std::vector<PsDataType> default_bins(global_max_dim);
      ps_splits->densePull(all_splits.data(), all_splits.size());
      ps_num_bins->densePull(num_bins.data(), num_bins.size());
      ps_default_bins->densePull(default_bins.data(), default_bins.size());
      std::vector<std::vector<float>> splits;
      for (size_t fid = 0, offset = 0; fid < global_max_dim; fid++) {
        size_t end = offset + static_cast<size_t>(num_bins[fid]);
        splits.push_back(std::vector<float>(
          all_splits.begin() + offset, 
          all_splits.begin() + end));
        offset = end;
      }
      feat_info = new FeatureInfo(splits, 
        std::vector<uint32_t>(num_bins.begin(), num_bins.end()), 
        std::vector<uint32_t>(default_bins.begin(), default_bins.end()));
    }
    return feat_info;
  }

  void Init(const Dataset<label_t, float>& train_data, 
            const Dataset<label_t, float>& valid_data) override {
    GBDTTrainer::Init(train_data, valid_data);
    uint32_t num_feat = this->feat_info->num_feat;
    uint32_t m1_size = this->ins_info->m1_size;
    uint32_t m2_size = this->ins_info->m2_size;
    if (this->param->IsMultiClassMultiTree())
      m1_size = m2_size = 1;
    uint32_t max_node_num = MAX_NODE_NUM(this->param->max_depth);
    uint32_t max_inner_num = MAX_INNER_NODE_NUM(this->param->max_depth);
    uint32_t layer_node_num = LAYER_NODE_NUM(this->param->max_depth - 1);
    this->ps_node_sizes.reset(new PSVector<PsDataType>(
      "node_sizes", max_node_num));
    this->ps_root_gp.reset(new PSVector<PsDataType>(
      "root_gp", m1_size + m2_size));
    int num_hist_bin = 0;
    for (uint32_t fid = 0; fid < num_feat; fid++) {
      if (!this->feat_info->splits[fid].empty()) {
        num_hist_bin += this->feat_info->num_bins[fid];
      }
    }
    this->ps_hist.reset(new PSMatrix<PsDataType>("hist", 
      num_hist_bin * (m1_size + m2_size), max_inner_num));
    this->ps_splits.reset(new PSMatrix<PsDataType>("splits", 
      3 + (m1_size + m2_size) * 2, max_inner_num));
  }

  /* -------------------- Training APIs -------------------- */

  void DoInitPreds(std::vector<float>& init_preds) override;
  
  std::tuple<uint32_t, GradPairPtr> 
  DoNewTree(int class_id = -1, uint64_t ins_sp_seed = 0, 
            uint64_t feat_sp_seed = 0) override;

  GBTSplitPtr DoFindRootSplit(const GradPair& root_gp, 
      const float root_gain, int class_id = -1) override;
  
  GBTSplitPtrMapPtr DoFindSplits(const std::vector<nid_t>& nids, 
                                 const std::vector<bool>& can_splits, 
                                 const std::vector<nid_t>& to_build, 
                                 const std::vector<nid_t>& to_subtract, 
                                 const std::vector<nid_t>& to_remove, 
                                 const std::vector<GradPairPtr>& node_sum_gps, 
                                 const std::vector<float>& node_gains, 
                                 int class_id = -1) override;
    
  std::map<uint32_t, uint32_t> DoSplitNodes(const GBTSplitPtrMap& splits, 
                                            int class_id = -1) override;


private:
  GBTSplitPtrMapPtr 
  MergeNodeHistsAndFindSplits(const std::vector<nid_t>& nids, 
                              const std::vector<GradPairPtr>& node_sum_gps, 
                              const std::vector<float>& node_gains);

  inline void GlobalSync() {
    PSAgent<PsDataType>::Get()->barrier();
  }

  inline void PushNodeHists(const std::vector<nid_t>& nids) {
    auto m1_size = this->ins_info->m1_size;
    auto m2_size = this->ins_info->m2_size;
    if (this->param->IsMultiClassMultiTree())
      m1_size = m2_size = 1;
    int num_hist_bin = 0, num_total_hist_bin = 0;
    for (uint32_t fid = 0; fid < this->feat_info->num_feat; fid++) {
      if (feat_info->is_feat_used[fid]) {
        num_hist_bin += this->feat_info->num_bins[fid];
      }
      if (!this->feat_info->splits[fid].empty()) {
        num_total_hist_bin += this->feat_info->num_bins[fid];
      }
    }
    std::vector<PsDataType> contents;
    // TODO: remove the placeholder for unsampled features
    // contents.reserve(nids.size() * num_hist_bin * (m1_size + m2_size));
    contents.resize(nids.size() * num_total_hist_bin * (m1_size + m2_size));
    auto offset = contents.begin();
    for (nid_t nid : nids) {
      const auto& node_hist = hist_manager->get(nid);
      for (uint32_t fid = 0; fid < this->feat_info->num_feat; fid++) {
        const auto& m1 = node_hist[fid]->get_m1();
        const auto& m2 = node_hist[fid]->get_m2();
        if (feat_info->is_feat_used[fid]) {
          std::copy(m1.begin(), m1.end(), offset);
          offset += m1.size();
          std::copy(m2.begin(), m2.end(), offset);
          offset += m2.size();
        }
      }
    }
    std::vector<int> col_ids(nids.begin(), nids.end());
    this->ps_hist->pushCols(col_ids.data(), contents.data(), 
      col_ids.size(), true);
  }

  inline GBTSplitPtrMapPtr 
  PullNodeHistsAndFindSplits(const std::vector<nid_t>& nids, 
                             const std::vector<GradPairPtr>& node_sum_gps, 
                             const std::vector<float>& node_gains) {
    std::vector<int> col_ids(nids.begin(), nids.end());
    auto m1_size = this->ins_info->m1_size;
    auto m2_size = this->ins_info->m2_size;
    if (this->param->IsMultiClassMultiTree())
      m1_size = m2_size = 1;
    int num_hist_bin = 0, num_total_hist_bin = 0;
    for (uint32_t fid = 0; fid < this->feat_info->num_feat; fid++) {
      if (feat_info->is_feat_used[fid]) {
        num_hist_bin += this->feat_info->num_bins[fid];
      }
      if (!this->feat_info->splits[fid].empty()) {
        num_total_hist_bin += this->feat_info->num_bins[fid];
      }
    }
    std::vector<PsDataType> contents;
    // TODO: remove the placeholder for unsampled features
    // contents.resize(nids.size() * num_hist_bin * (m1_size + m2_size));
    contents.resize(nids.size() * num_total_hist_bin * (m1_size + m2_size));
    this->ps_hist->pullCols(col_ids.data(), contents.data(), 
      col_ids.size(), true);
    std::vector<double> m1, m2;
    auto offset = contents.begin();
    // TODO: server-side split finding
    GBTSplitPtrMapPtr splits(new GBTSplitPtrMap());
    for (uint32_t i = 0; i < nids.size(); i++) {
      std::vector<GBTSplitPtr> buffer(this->feat_info->num_feat);
      for (uint32_t fid = 0; fid < this->feat_info->num_feat; fid++) {
        if (feat_info->is_feat_used[fid]) {
          m1.resize(feat_info->num_bins[fid] * m1_size);
          m2.resize(feat_info->num_bins[fid] * m2_size);
          std::copy(offset, offset + m1.size(), m1.begin());
          offset += m1.size();
          std::copy(offset, offset + m2.size(), m2.begin());
          offset += m2.size();
          buffer[fid] = split_finder->FindBestSplitPointOfOneFeature(
            m1, m1_size, m2, m2_size, 
            fid, feat_info->num_bins[fid], feat_info->splits[fid], 
            *node_sum_gps[i], node_gains[i], *tree_booster);
        }
      }
      // choose the best split among all features
      int best_fid = -1;
      float best_gain = 0;
      for (int fid = 0; fid < buffer.size(); fid++) {
        if (buffer[fid] != nullptr) {
          float gain = buffer[fid]->get_split_entry().get_gain();
          if (gain > best_gain) {
            best_fid = fid;
            best_gain = gain;
          }
        }
      }
      if (best_fid != -1 && buffer[best_fid] != nullptr) {
        splits->insert(std::make_pair(nids[i], buffer[best_fid]));
      }
    }   
    return splits;
  }

  inline void PushNodeSplits(const GBTSplitPtrMapPtr splits) {
    uint32_t m1_size = this->ins_info->m1_size;
    uint32_t m2_size = this->ins_info->m2_size;
    if (this->param->IsMultiClassMultiTree())
      m1_size = m2_size = 1;
    uint32_t num_row = 3 + (m1_size + m2_size) * 2;
    std::vector<nid_t> nids;
    nids.reserve(splits->size());
    std::vector<PsDataType> contents(splits->size() * num_row);
    for (const auto& node_split : *splits) {
      auto offset = nids.size() * num_row;
      nid_t nid = node_split.first;
      nids.push_back(nid);
      const auto& split_point = static_cast<const SplitPoint&>(
        node_split.second->get_split_entry());
      contents[offset] = split_point.get_fid();
      contents[offset + 1] = split_point.get_value();
      contents[offset + 2] = split_point.get_gain();
      node_split.second->get_left_gp().get(
        contents.data() + offset + 3);
      node_split.second->get_right_gp().get(
        contents.data() + offset + 3 + m1_size + m2_size);
    }

    std::vector<int> col_ids(nids.begin(), nids.end());
    this->ps_splits->pushCols(col_ids.data(), contents.data(), 
      col_ids.size(), true);
  }

  inline GBTSplitPtrMapPtr PullNodeSplits(const std::vector<nid_t>& nids) {
    GBTSplitPtrMapPtr splits(new GBTSplitPtrMap());
    std::vector<int> col_ids(nids.begin(), nids.end());
    uint32_t m1_size = this->ins_info->m1_size;
    uint32_t m2_size = this->ins_info->m2_size;
    if (this->param->IsMultiClassMultiTree())
      m1_size = m2_size = 1;
    uint32_t num_row = 3 + (m1_size + m2_size) * 2;
    std::vector<PsDataType> contents(nids.size() * num_row);
    this->ps_splits->pullCols(col_ids.data(), contents.data(), 
      col_ids.size(), true);
    for (uint32_t i = 0; i < nids.size(); i++) {
      nid_t nid = nids[i];
      auto offset = i * num_row;
      int fid = contents[offset];
      float value = contents[offset + 1];
      float gain = contents[offset + 2];
      if (gain != 0 && gain > this->param->min_split_gain) {
        auto* split_point = new SplitPoint(fid, value, gain);
        auto* left_gp = this->node_indexer->get_node_gp(0).zeros_like();
        left_gp->set(contents.data() + offset + 3);
        auto* right_gp = left_gp->zeros_like();
        right_gp->set(contents.data() + offset + 3 + m1_size + m2_size);
        auto node_split = GBTSplitPtr(new GBTSplit(
          split_point, left_gp, right_gp));
        splits->insert(std::make_pair(nid, node_split));
      }
    }
    return splits;
  }

  int rank;
  int num_workers;
  PsVectorPtr ps_node_sizes;
  PsVectorPtr ps_root_gp;
  PsMatrixPtr ps_hist;
  PsMatrixPtr ps_splits;
  int xxx = 0;
};

void GBDTDPTrainer::DoInitPreds(std::vector<float>& init_preds) {
  if (param->is_regression) 
    return;

  PsVectorPtr ps_labels_cnts(new PSVector<PsDataType>(
    "label_cnts", param->num_label));
  if (this->rank == 0) {
    ps_labels_cnts->initAllZeros();
  }
  GlobalSync();

  // count and summarize label occurences
  std::vector<uint32_t> cnts;
  CheckAndCountLabels(train_data->get_labels(), cnts, param->num_label);
  std::vector<PsDataType> sum_cnts(cnts.begin(), cnts.end());
  ps_labels_cnts->densePush(sum_cnts.data(), param->num_label);
  GlobalSync();
  ps_labels_cnts->densePull(sum_cnts.data(), param->num_label);
  PsDataType num_ins = 0;
  for (auto cnt : sum_cnts) 
    num_ins += cnt;
  double avg = 1.0 / param->num_label;
  if (param->num_label == 2) {
    PsDataType num_neg = sum_cnts[0], num_pos = sum_cnts[1];
    HML_LOG_INFO << "Rank[" << this->rank << "] Labels: negatives[" 
      << num_neg << "] positives[" << num_pos << "]";
    init_preds.resize(1);
    init_preds[0] = (float) (((double) num_pos) / ((double) num_ins) - avg);
    init_preds[0] *= param->learning_rate;
    // update preds
    std::fill(ins_info->predictions.begin(), 
      ins_info->predictions.end(), init_preds[0]);
    std::fill(valid_preds.begin(), valid_preds.end(), init_preds[0]);
  } else {
    HML_LOG_INFO << "Rank[" << this->rank << "] Label occurrences: " << sum_cnts;
    init_preds.resize(param->num_label);
    for (int k = 0; k < param->num_label; k++) {
      init_preds[k] = (float) (sum_cnts[k] / ((double) num_ins) - avg);
      init_preds[k] *= param->learning_rate;
    }
    // update preds
    for (uint32_t i = 0; i < num_train; i++) {
      std::copy(init_preds.begin(), init_preds.end(), 
        ins_info->predictions.begin() + i * param->num_label);
    }
    for (uint32_t i = 0; i < num_valid; i++) {
      std::copy(init_preds.begin(), init_preds.end(), 
        valid_preds.begin() + i * param->num_label);
    }
  }
}

std::tuple<uint32_t, GradPairPtr> 
GBDTDPTrainer::DoNewTree(int class_id, uint64_t ins_sp_seed, 
                         uint64_t feat_sp_seed) {
  if (this->rank == 0) {
    // init ps vector
    this->ps_node_sizes->initAllZeros();
    this->ps_root_gp->initAllZeros();
    this->ps_hist->initAllZeros();
    this->ps_splits->initAllZeros();
  }
  GlobalSync();
  // persist root grad pairs of worker
  GradPairPtr root_gp = node_indexer->get_node_gp_ptr(0);
  // reset node indexing
  node_indexer->Reset(*ins_info);
  // instance sampling & reset root gp of worker
  if (tree_sampler->InsSampling(*ins_info, *node_indexer, 
        class_id, ins_sp_seed)) {
    auto num_sampled = node_indexer->get_node_size(0);
    auto ratio = 1.0 * num_sampled / node_indexer->get_num_ins();
    HML_LOG_INFO << "Rank[" << this->rank << "] Sampled " << num_sampled 
      << " instances, ratio = " << ratio;
    node_indexer->SumNodeGradPairs(0, *ins_info, *param, class_id);
  } else if (param->IsMultiClassMultiTree()) {
    node_indexer->SumNodeGradPairs(0, *ins_info, *param, class_id);
  } else {
    node_indexer->set_node_gp_ptr(0, root_gp);
  }
  // summarize number of sampled instance and root gp on PS
  int root_id = 0; PsDataType root_size = node_indexer->get_node_size(0);
  this->ps_node_sizes->sparsePush(&root_id, &root_size, 1, true);
  std::vector<PsDataType> root_gp_vec;
  root_gp->get(root_gp_vec);
  this->ps_root_gp->densePush(root_gp_vec.data(), root_gp_vec.size());
  // feature sampling, TODO: sync feat_sp_seed among workers
  if (tree_sampler->FeatSampling(*feat_info, feat_sp_seed) 
      || ins_info->m2_size != param->m2_size()) {
    hist_manager->Reset();
  }
  // pull global statistics from PS
  GlobalSync();
  root_id = 0;
  this->ps_node_sizes->sparsePull(&root_id, &root_size, 1, true);
  uint32_t global_root_size = static_cast<uint32_t>(root_size);
  GradPairPtr global_root_gp(root_gp->zeros_like());
  this->ps_root_gp->densePull(root_gp_vec.data(), root_gp_vec.size());
  global_root_gp->set(root_gp_vec);
  // HML_LOG_INFO << "Rank[" << this->rank << "] Root size: " 
  //   << "local[" << node_indexer->get_node_size(0) << "] "
  //   << "global[" << global_root_size << "]";
  // HML_LOG_INFO << "Rank[" << this->rank << "] Root grad pair: " 
  //   << "local[" << node_indexer->get_node_gp(0) << "] "
  //   << "global[" << *global_root_gp << "]";

  return std::tuple<uint32_t, GradPairPtr>(
    global_root_size, global_root_gp);
}

GBTSplitPtr GBDTDPTrainer::DoFindRootSplit(const GradPair& root_sum_gp, 
    const float root_gain, int class_id) {
  TIK(build);
  hist_manager->BuildHistForRoot(*train_data, 
    *ins_info, *feat_info, *node_indexer, class_id);
  TOK(build);
  HML_LOG_DEBUG << "Rank[" << this->rank << "] Build hist for root" 
    << " cost " << COST_MSEC(build) << " ms";
  TIK(find);
  GBTSplitPtrMapPtr splits = MergeNodeHistsAndFindSplits(
    {0}, {GradPairPtr(root_sum_gp.copy())}, {root_gain});
  GBTSplitPtr root_split = nullptr;
  if (!splits->empty()) root_split = (*splits->find(0)).second;
  TOK(find);
  HML_LOG_DEBUG << "Rank[" << this->rank << "] Merge histograms and " 
    << "find split for root cost " << COST_MSEC(find) << " ms";
  return root_split;
}

GBTSplitPtrMapPtr GBDTDPTrainer::DoFindSplits(const std::vector<nid_t>& nids, 
    const std::vector<bool>& can_splits, const std::vector<nid_t>& to_build, 
    const std::vector<nid_t>& to_subtract, const std::vector<nid_t>& to_remove, 
    const std::vector<GradPairPtr>& node_sum_gps, 
    const std::vector<float>& node_gains, int class_id) {
  // remove hist of nodes whose children cannot be split
  for (nid_t nid : to_remove) hist_manager->remove(nid);
  // build hist for nodes with fewer instances
  TIK(build);
  hist_manager->BuildHistForNodes(to_build, *train_data, 
    *ins_info, *feat_info, *node_indexer, class_id);
  TOK(build);
  HML_LOG_DEBUG << "Rank[" << this->rank << "] Build hist for nodes " 
    << to_build << " cost " << COST_MSEC(build) << " ms";
  // subtract hist for nodes with more instances
  // TODO: perform subtraction on PS
  TIK(hist_sub);
  for (nid_t nid : to_subtract) hist_manager->HistSubtract(nid);
  TOK(hist_sub);
  HML_LOG_DEBUG << "Rank[" << this->rank << "] Subtract hist for nodes " 
    << to_subtract << " cost " << COST_MSEC(hist_sub) << " ms";
  // merge built histograms and find best splits
  TIK(find);
  std::vector<nid_t> to_find_nids;
  std::vector<GradPairPtr> to_find_node_sum_gps;
  std::vector<float> to_find_node_gains;
  for (int i = 0; i < nids.size(); i++) {
    nid_t nid = nids[i];
    if (can_splits[i]) {
      to_find_nids.push_back(nid);
      to_find_node_sum_gps.push_back(node_sum_gps[i]);
      to_find_node_gains.push_back(node_gains[i]);
    } else {
      // node which cannot be split, but we build hist 
      // for it to perform hist subtraction, remove here
      hist_manager->remove(nid);
    }
  }
  GBTSplitPtrMapPtr splits = MergeNodeHistsAndFindSplits(to_find_nids, 
    to_find_node_sum_gps, to_find_node_gains);  
  TOK(find);
  HML_LOG_DEBUG << "Rank[" << this->rank << "] Merge histograms and " 
    << "find splits for nodes " << nids 
    << " cost " << COST_MSEC(find) << " ms";
  return splits;
}

std::map<uint32_t, uint32_t> GBDTDPTrainer::DoSplitNodes(
    const GBTSplitPtrMap& splits, int class_id) {
  TIK(update_pos);
  std::vector<nid_t> parents;
  for (auto const& t : splits) {
    nid_t nid = t.first;
    parents.push_back(nid);
    auto& split = t.second;
    auto& split_entry = split->get_split_entry();
    node_indexer->UpdatePos(nid, *train_data, split_entry, 
      feat_info->splits[split_entry.get_fid()]);
    node_indexer->SumNodeGradPairs(2 * nid + 1, *ins_info, *param, class_id);
    node_indexer->SumNodeGradPairs(2 * nid + 2, *ins_info, *param, class_id);
  }
  std::sort(parents.begin(), parents.end());
  TOK(update_pos);
  HML_LOG_DEBUG << "Rank[" << this->rank << "] " 
    << "Update instance postition for nodes " << parents
    << " cost " << COST_MSEC(update_pos) << " ms";
  // summarize node sizes on PS
  TIK(sum_size);
  std::vector<int> children_nids;
  std::vector<PsDataType> chilren_sizes;
  for (nid_t nid : parents) {
    children_nids.push_back(2 * nid + 1);
    children_nids.push_back(2 * nid + 2);
    chilren_sizes.push_back(node_indexer->get_node_size(2 * nid + 1));
    chilren_sizes.push_back(node_indexer->get_node_size(2 * nid + 2));
  }
  this->ps_node_sizes->sparsePush(children_nids.data(), 
    chilren_sizes.data(), children_nids.size(), false);
  GlobalSync();
  this->ps_node_sizes->sparsePull(children_nids.data(), 
    chilren_sizes.data(), children_nids.size(), false);
  std::map<uint32_t, uint32_t> chilren_sizes_map;
  for (uint32_t i = 0; i < children_nids.size(); i++) {
    chilren_sizes_map[children_nids[i]] = chilren_sizes[i];
    // HML_LOG_DEBUG << "Rank[" << this->rank << "] " 
    //   << "Size of node[" << children_nids[i] << "]: " 
    //   << "local[" << node_indexer->get_node_size(children_nids[i]) << "], "
    //   << "global[" << chilren_sizes[i] << "]";
  }
  TOK(sum_size);
  HML_LOG_DEBUG << "Rank[" << this->rank << "] Sum sizes of children nodes"
    << " cost " << COST_MSEC(sum_size) << " ms";
  
  return chilren_sizes_map;
}

GBTSplitPtrMapPtr GBDTDPTrainer::MergeNodeHistsAndFindSplits
(const std::vector<nid_t>& nids, 
 const std::vector<GradPairPtr>& node_sum_gps, 
 const std::vector<float>& node_gains) {
  // push node hist
  TIK(push_hist);
  PushNodeHists(nids);
  TOK(push_hist);
  // HML_LOG_DEBUG << "Rank[" << this->rank << "] " 
  //   << "Push histograms of nodes " << nids 
  //   << " cost " << COST_MSEC(push_hist) << " ms";
  GlobalSync();

  // pull node hist & find split with load balancing
  std::vector<nid_t> sharded_nids;
  std::vector<GradPairPtr> sharded_node_sum_gps;
  std::vector<float> sharded_node_gains;
  for (uint32_t i = 0; i < nids.size(); i++) {
    if (i % this->num_workers == this->rank) {
      sharded_nids.push_back(nids[i]);
      sharded_node_sum_gps.push_back(node_sum_gps[i]);
      sharded_node_gains.push_back(node_gains[i]);
    }
  }
  if (!sharded_nids.empty()) {
    TIK(find_splits);
    auto sharded_splits = PullNodeHistsAndFindSplits(
      sharded_nids, sharded_node_sum_gps, sharded_node_gains);
    TOK(find_splits);
    // HML_LOG_DEBUG << "Rank[" << this->rank << "] " 
    //   << "Pull histograms and find splits of nodes " << sharded_nids 
    //   << " cost " << COST_MSEC(push_hist) << " ms";

    // push splits that the worker found
    TIK(push_splits);
    PushNodeSplits(sharded_splits);
    TOK(push_splits);
  }

  // pull splits for all nodes
  GlobalSync();
  TIK(pull_splits);
  auto splits = PullNodeSplits(nids);
  TOK(pull_splits);
  
  return splits;
}

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_TRAIN_PARALLEL_TRAINER_H_
