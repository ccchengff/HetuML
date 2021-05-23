#include "model/gbdt/train/trainer.h"

namespace hetu { 
namespace ml {
namespace gbdt {

GBDTModel* GBDTTrainer::FitModel() {
  HML_LOG_INFO << "Start to fit model...";
  GBDTModel* model = new GBDTModel(this->param);
  TIK(train);

  // tree/node buffers
  int num_trees_per_round = param->IsMultiClassMultiTree() 
    ? param->num_label : 1;
  int max_node_num = MAX_NODE_NUM(param->max_depth);
  int max_inner_num = MAX_INNER_NODE_NUM(param->max_depth);
  GBTNodePtrVec node_buffer(max_node_num);
  GBTSplitPtrVec split_buffer(max_inner_num);
  std::vector<nid_t> to_find, to_split, to_set_leaves;
  to_find.reserve(max_node_num);
  to_split.reserve(max_node_num);
  to_set_leaves.reserve(max_node_num);

  // initialize predictions
  if (!this->param->is_majority_voting)
    InitPreds(*model);

  for (int round_id = 0; round_id < param->num_round; round_id++) {
    // DT_LOG_INFO("Start to train round[" << round_id + 1 << "]");
    TIK(cur_round);

    // calc grad pairs for current round 
    NewRound(round_id);
    // train trees in current round
    for (int tree_id = 0; tree_id < num_trees_per_round; tree_id++) {
      TIK(cur_tree);

      // reset buffers
      std::fill(node_buffer.begin(), node_buffer.end(), nullptr);
      std::fill(split_buffer.begin(), split_buffer.end(), nullptr);
      to_find.clear();
      to_split.clear();
      to_set_leaves.clear();
      int class_id = (num_trees_per_round > 1) ? tree_id : -1;

      // create new tree
      GBTreePtr tree(new GBTree(param->max_depth));
      NewTree(node_buffer, class_id);
      // iteratively build one tree
      int cur_node_num = 1;
      to_find.push_back(0);
      while (cur_node_num + 2 < param->max_node_num) {
        // build histograms and find splits
        if (!to_find.empty()) {
          if (to_find[0] == 0) {
            auto split = FindRootSplit(*node_buffer[0], class_id);
            if (split != nullptr) {
              split_buffer[0].swap(split);
            } else {
              HML_LOG_WARN << "Root cannot be split";
            }
          } else {
            // sort to make sure behaviors of all workers are the same
            if (to_find.size() > 1)
              std::sort(to_find.begin(), to_find.end());
            FindSplits(to_find, node_buffer, split_buffer, class_id);
          }
        } 
        // choose splits
        ChooseSplits(to_find, to_split, to_set_leaves, 
          split_buffer, cur_node_num);
        if (to_find.empty() && to_split.empty()) {
          HML_LOG_DEBUG << "No more splits for current tree";
          break;
        }
        to_find.clear();
        // split nodes
        if (!to_split.empty()) {
          if (to_split.size() > 1) 
            std::sort(to_split.begin(), to_split.end());
          SplitNodes(*tree, to_split, node_buffer, split_buffer, class_id);
          for (nid_t nid : to_split) {
            if (2 * nid + 1 < max_inner_num) {
              to_find.push_back(2 * nid + 1);
              to_find.push_back(2 * nid + 2);
            } else {
              to_set_leaves.push_back(2 * nid + 1);
              to_set_leaves.push_back(2 * nid + 2);
            }
          }
          cur_node_num += 2 * to_split.size();
          to_split.clear();
        }
        // set leaves
        if (!to_set_leaves.empty()) {
          if (to_set_leaves.size() > 1)
            std::sort(to_set_leaves.begin(), to_set_leaves.end());
          SetAsLeaves(*tree, to_set_leaves, node_buffer);
          to_set_leaves.clear();
        }
      }
      // finish tree & update predictions
      FinishTree(*tree, node_buffer);
      ASSERT(tree->get_num_nodes() == cur_node_num) 
        << "#nodes not matching: expected " << cur_node_num 
        << " but got " << tree->get_num_nodes();
      UpdatePreds(*tree, param->learning_rate, class_id);
      TOK(cur_tree);
      TOK(train);
      HML_LOG_INFO << "Train tree[" << tree_id + 1 << "/" 
        << num_trees_per_round  << "] of round[" << round_id + 1 
        << "] with " << tree->get_num_nodes() 
        << " nodes (" << tree->get_num_leaves() << " leaves)" 
        << " cost " << COST_MSEC(cur_tree) << " ms, " 
        << COST_MSEC(train) << " ms elapsed";
      model->add(tree);
    }
    // evaluation
    Evaluate(round_id);
    // finish training of current round
    TOK(cur_round);
    TOK(train);
    HML_LOG_INFO << "Train round[" << round_id + 1 << "]" 
      << " cost " << COST_MSEC(cur_round) << " ms, "
      << COST_MSEC(train) << " ms elapsed";
  }

  TOK(train);
  HML_LOG_INFO << "Fit model with " << model->num_trees() << " tree(s)"
    << " cost " << COST_MSEC(train) << " ms";
  return model;
}

void GBDTTrainer::InitPreds(GBDTModel& model) {
  TIK(init_preds);
  std::vector<float> init_preds(param->pred_size());
  DoInitPreds(init_preds); 
  model.set_init_preds(init_preds);
  TOK(init_preds);
  HML_LOG_DEBUG << "Initialize predictions cost " 
    << COST_MSEC(init_preds) << " ms";
}

void GBDTTrainer::NewRound(int round_id) {
  if (!this->param->is_majority_voting || round_id == 0) {
    TIK(new_rond);
    DoNewRound();
    TOK(new_rond);
    HML_LOG_DEBUG << "New round[" << round_id 
      << "] calc grad pairs cost "
      << COST_MSEC(new_rond) << " ms";
  }
}

void GBDTTrainer::NewTree(GBTNodePtrVec& node_buffer, int class_id) {
  TIK(new_tree);
  GBTNodePtr root(new GBTNode(0));
  auto root_meta = DoNewTree(class_id);
  root->set_size(std::get<0>(root_meta));
  root->set_sum_gp(*(std::get<1>(root_meta)));
  tree_booster->CalcNodeGain(*root);
  node_buffer[0].swap(root);
  TOK(new_tree);
  HML_LOG_DEBUG << "New tree cost " << COST_MSEC(new_tree) << " ms";
}

GBTSplitPtr GBDTTrainer::FindRootSplit(const GBTNode& root, int class_id) {
  TIK(find_root);
  GBTSplitPtr ret = nullptr;
  HML_LOG_DEBUG << "Root meta: gp[" << root.get_sum_gp() 
    << "], gain[" << root.get_gain() << "]";
  if (NodeCanSplit(root)) 
    ret = DoFindRootSplit(root.get_sum_gp(), root.get_gain(), class_id);
  TOK(find_root);
  HML_LOG_DEBUG << "Find best split of root node"
    << " cost " << COST_MSEC(find_root) << " ms";
  if (ret != nullptr) 
    HML_LOG_DEBUG << "Best split of root: " << *ret;
  else 
    HML_LOG_DEBUG << "Root node does not have a split";
  return ret;
}

void GBDTTrainer::FindSplits(const std::vector<nid_t>& to_find, 
                             GBTNodePtrVec& node_buffer, 
                             GBTSplitPtrVec& split_buffer, int class_id) {
  TIK(find_splits);

  ASSERT(IS_EVEN(to_find.size())) << "Number of active nodes should be even";
  std::vector<bool> can_splits(to_find.size());
  std::vector<GradPairPtr> node_sum_gps(to_find.size());
  std::vector<float> node_gains(to_find.size());
  for (int i = 0; i < to_find.size(); i++) {
    auto& node = *node_buffer[to_find[i]];
    can_splits[i] = NodeCanSplit(node);
    if (can_splits[i]) {
      // TODO: do not copy gp here
      node_sum_gps[i].reset(node.get_sum_gp().copy());
      node_gains[i] = node.get_gain();
    }
  }
  
  std::vector<nid_t> to_build, to_subtract, to_remove;
  for (int i = 0; i < to_find.size(); i += 2) {
    nid_t l = to_find[i], r = to_find[i + 1];
    ASSERT(l + 1 == r) << "Sibling nodes should be processes together";
    can_splits[i] = NodeCanSplit(*node_buffer[l]);
    can_splits[i + 1] = NodeCanSplit(*node_buffer[r]);
    if (!can_splits[i] && !can_splits[i + 1]) {
      // remove histograms of parent node
      to_remove.push_back(PARENT(l));
    } else {
      // build histograms for one & subtract for another if needed
      if (node_buffer[l]->get_size() < node_buffer[r]->get_size()) {
        to_build.push_back(l);
        if (can_splits[i + 1]) to_subtract.push_back(r);
      } else {
        to_build.push_back(r);
        if (can_splits[i]) to_subtract.push_back(l);
      }
    }
  }

  GBTSplitPtrMapPtr splits = DoFindSplits(to_find, can_splits, 
    to_build, to_subtract, to_remove, node_sum_gps, node_gains, 
    class_id);
  for (auto & split : *splits) {
    nid_t nid = split.first;
    GBTSplitPtr& node_split = split.second;
    split_buffer[nid].swap(node_split);
    HML_LOG_DEBUG << "Best split of node[" << nid 
      << "]: " << *split_buffer[nid];
  }
  
  TOK(find_splits);
  HML_LOG_DEBUG << "Find best splits cost " << COST_MSEC(find_splits) << " ms";
}

void GBDTTrainer::ChooseSplits(std::vector<nid_t>& to_find, 
                               std::vector<nid_t>& to_split, 
                               std::vector<nid_t>& to_set_leaves, 
                               GBTSplitPtrVec& split_buffer, 
                               int cur_node_num) {
  // if split is nullptr, then set as leaf
  for (nid_t nid : to_find) {
    if (split_buffer[nid] == nullptr)
      to_set_leaves.push_back(nid);
  }
  // extract all nodes with splits
  std::vector<std::tuple<float, nid_t>> split_gains;
  for (nid_t nid = 0; nid < split_buffer.size(); nid++) {
    if (split_buffer[nid] != nullptr) {
      float gain = split_buffer[nid]->get_split_entry().get_gain();
      split_gains.push_back(std::make_tuple(-gain, nid));
    }
  }
  if (split_gains.empty()) 
    return;
  // determine which splits can be executed
  if (!param->leafwise) {
    int num_nodes_left = (param->max_node_num - cur_node_num) / 2;
    if (num_nodes_left >= split_gains.size()) {
      // all splits can be executed
      for (auto & t : split_gains) 
        to_split.push_back(std::get<1>(t));
    } else {
      // choose splits with largest gains
      sort(split_gains.begin(), split_gains.end());
      for (int i = 0; i < num_nodes_left; i++) 
        to_split.push_back(std::get<1>(split_gains[i]));
      for (int i = num_nodes_left; i < split_gains.size(); i++)
        to_set_leaves.push_back(std::get<1>(split_gains[i]));
    }
  } else {
    // choose one split with largest gain
    auto it = std::max_element(split_gains.begin(), split_gains.end());
    nid_t max_nid = std::get<1>(*it);
    to_split.push_back(max_nid);
    if (cur_node_num + 2 == param->max_node_num) {
      // set the remaining as leaves
      for (auto & t : split_gains) {
        nid_t nid = std::get<0>(t);
        if (nid != max_nid)
          to_set_leaves.push_back(nid);
      }
    }
  }
}

void GBDTTrainer::SplitNodes(GBTree& tree, 
                             const std::vector<nid_t>& to_split, 
                             GBTNodePtrVec& node_buffer, 
                             GBTSplitPtrVec& split_buffer, 
                             int class_id) {
  TIK(split_nodes);

  GBTSplitPtrMap splits;
  for (nid_t nid : to_split) splits[nid] = split_buffer[nid];
  auto children_sizes = DoSplitNodes(splits, class_id);
  for (nid_t nid : to_split) {
    // set node split entry, put node to tree
    auto& node = node_buffer[nid];
    auto node_split = split_buffer[nid];
    node->set_split_entry(node_split->get_split_entry());
    tree.set_node(nid, node);
    node_buffer[nid] = nullptr;
    split_buffer[nid] = nullptr;
    // create left child node
    GBTNodePtr left_child(new GBTNode(2 * nid + 1));
    left_child->set_size(children_sizes[2 * nid + 1]);
    left_child->set_sum_gp(node_split->get_left_gp());
    tree_booster->CalcNodeGain(*left_child);
    node_buffer[2 * nid + 1].swap(left_child);
    // create right child node
    GBTNodePtr right_child(new GBTNode(2 * nid + 2));
    right_child->set_size(children_sizes[2 * nid + 2]);
    right_child->set_sum_gp(node_split->get_right_gp());
    tree_booster->CalcNodeGain(*right_child);
    node_buffer[2 * nid + 2].swap(right_child);
  }

  TOK(split_nodes);
  HML_LOG_DEBUG << "Split nodes " << to_split 
    << " cost " << COST_MSEC(split_nodes) << " ms";
}

void GBDTTrainer::SetAsLeaves(GBTree& tree, 
                              const std::vector<nid_t>& to_set_leaves, 
                              GBTNodePtrVec& node_buffer) {
  DoSetAsLeaves(to_set_leaves);
  for (nid_t nid : to_set_leaves) {
    auto& leaf = node_buffer[nid];
    leaf->chg_to_leaf();
    if (!param->IsLeafVector())
      tree_booster->CalcLeafWeight(*leaf);
    else 
      tree_booster->CalcLeafWeights(*leaf);
    tree.set_node(nid, leaf);
    node_buffer[nid] = nullptr;
  }
}

void GBDTTrainer::FinishTree(GBTree& tree, GBTNodePtrVec& node_buffer) {
  std::vector<nid_t> leaves;
  for (nid_t nid = 0; nid < node_buffer.size(); nid++) {
    if (node_buffer[nid] != nullptr) 
      leaves.push_back(nid);
  }
  if (!leaves.empty())
    SetAsLeaves(tree, leaves, node_buffer);
}

void GBDTTrainer::UpdatePreds(GBTree& tree, 
    float learning_rate, int class_id) {
  TIK(update_preds);
  DoUpdatePreds(tree, learning_rate, class_id);
  TOK(update_preds);
  HML_LOG_DEBUG << "Update predictions" 
    << " cost " << COST_MSEC(update_preds) << " ms";
}

void GBDTTrainer::Evaluate(int round_id) {
  TIK(evaluate);
  auto metrics = DoEvaluate(round_id);
  std::ostringstream train_os, valid_os;
  for (auto const& metric : metrics) {
    train_os << metric.first << "[" << std::get<0>(metric.second) << "] ";
  }
  HML_LOG_INFO << "Evaluation on train data of round[" 
    << round_id + 1 << "]: " << train_os.str();
  if (this->num_valid > 0) {
    for (auto const& metric : metrics)
      valid_os << metric.first << "[" << std::get<1>(metric.second) << "] ";
    HML_LOG_INFO << "Evaluation on valid data of round[" 
      << round_id + 1 << "]: " << valid_os.str();
  }
  TOK(evaluate);
  HML_LOG_DEBUG << "Evaluation of round[" << round_id + 1 << "]" 
    << " cost " << COST_MSEC(evaluate) << " ms";
}

/* -------------------- Impl for Standalone training -------------------- */

void GBDTTrainer::DoInitPreds(std::vector<float>& init_preds) {
  if (param->is_regression) 
    return;

  std::vector<uint32_t> cnts;
  CheckAndCountLabels(train_data->get_labels(), cnts, param->num_label);
  const uint32_t num_ins = train_data->get_num_instances();
  double avg = 1.0 / param->num_label;
  if (param->num_label == 2) {
    uint32_t num_neg = cnts[0], num_pos = cnts[1];
    HML_LOG_INFO << "Labels: negatives[" << num_neg 
      << "] positives[" << num_pos << "]";
    init_preds.resize(1);
    init_preds[0] = (float) (((double) num_pos) / ((double) num_ins) - avg);
    init_preds[0] *= param->learning_rate;
    HML_LOG_DEBUG << "init pred = " << init_preds[0];
    // update preds
    std::fill(ins_info->predictions.begin(), 
      ins_info->predictions.end(), init_preds[0]);
    std::fill(valid_preds.begin(), valid_preds.end(), init_preds[0]);
  } else {
    HML_LOG_INFO << "Label occurrences: " << cnts;
    init_preds.resize(param->num_label);
    for (int k = 0; k < param->num_label; k++) {
      init_preds[k] = (float) (((double) cnts[k]) / ((double) num_ins) - avg);
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

void GBDTTrainer::DoNewRound() {
  const auto* loss_func = \
    LossFactory::GetRichLoss<float, float, double>(this->param->loss);
  GradPairPtr root_gp(tree_booster->CalcGradPairs(
    *loss_func, train_data->get_labels(), *ins_info));
  node_indexer->set_node_gp_ptr(0, root_gp);
}

std::tuple<uint32_t, GradPairPtr> 
GBDTTrainer::DoNewTree(int class_id, uint64_t ins_sp_seed, 
                       uint64_t feat_sp_seed) {
  // persist root grad pairs
  GradPairPtr root_gp = node_indexer->get_node_gp_ptr(0);
  // reset node indexing
  node_indexer->Reset(*ins_info);
  // instance sampling & reset root gp
  if (tree_sampler->InsSampling(*ins_info, *node_indexer, 
        class_id, ins_sp_seed)) {
    auto num_sampled = node_indexer->get_node_size(0);
    auto ratio = 1.0 * num_sampled / node_indexer->get_num_ins();
    HML_LOG_INFO << "Sampled " << num_sampled 
      << " instances, ratio = " << ratio;
    node_indexer->SumNodeGradPairs(0, *ins_info, *param, class_id);
  } else if (param->IsMultiClassMultiTree()) {
    node_indexer->SumNodeGradPairs(0, *ins_info, *param, class_id);
  } else {
    node_indexer->set_node_gp_ptr(0, root_gp);
  }
  // feature sampling
  if (tree_sampler->FeatSampling(*feat_info, feat_sp_seed) 
      || ins_info->m2_size != param->m2_size()) {
    hist_manager->Reset();
  }

  return std::tuple<uint32_t, GradPairPtr>(
    node_indexer->get_node_size(0), node_indexer->get_node_gp_ptr(0));
}

GBTSplitPtr GBDTTrainer::DoFindRootSplit(const GradPair& root_sum_gp, 
                                         const float root_gain, 
                                         int class_id) {
  TIK(build);
  hist_manager->BuildHistForRoot(*train_data, 
    *ins_info, *feat_info, *node_indexer, class_id);
  TOK(build);
  HML_LOG_DEBUG << "Build hist for root cost " << COST_MSEC(build) << " ms";
  TIK(find);
  auto root_split = split_finder->FindBestSplit(*feat_info, 
    hist_manager->get(0), root_sum_gp, root_gain, *tree_booster);
  TOK(find);
  HML_LOG_DEBUG << "Find split for root cost " << COST_MSEC(find) << " ms";
  return root_split;
}

GBTSplitPtrMapPtr 
GBDTTrainer::DoFindSplits(const std::vector<nid_t>& nids, 
                          const std::vector<bool>& can_splits, 
                          const std::vector<nid_t>& to_build, 
                          const std::vector<nid_t>& to_subtract, 
                          const std::vector<nid_t>& to_remove, 
                          const std::vector<GradPairPtr>& node_sum_gps, 
                          const std::vector<float>& node_gains, 
                          int class_id) {
  // remove hist of nodes whose children cannot be split
  for (nid_t nid : to_remove) hist_manager->remove(nid);
  // build hist for nodes with fewer instances
  TIK(build);
  hist_manager->BuildHistForNodes(to_build, *train_data, 
    *ins_info, *feat_info, *node_indexer, class_id);
  TOK(build);
  HML_LOG_DEBUG << "Build hist for nodes " << to_build 
    << " cost " << COST_MSEC(build) << " ms";
  // subtract hist for nodes with more instances
  TIK(hist_sub);
  for (nid_t nid : to_subtract) hist_manager->HistSubtract(nid);
  TOK(hist_sub);
  HML_LOG_DEBUG << "Subtract hist for nodes " << to_subtract 
    << " cost " << COST_MSEC(hist_sub) << " ms";
  // find best splits
  TIK(find);
  GBTSplitPtrMapPtr splits(new GBTSplitPtrMap());
  for (int i = 0; i < nids.size(); i++) {
    nid_t nid = nids[i];
    if (can_splits[i]) {
      // find split if the node can be split
      auto split = split_finder->FindBestSplit(*feat_info, 
        hist_manager->get(nid), *node_sum_gps[i], 
        node_gains[i], *tree_booster);
      if (split != nullptr)
        splits->insert(std::make_pair(nid, split));
    } else {
      // node which cannot be split, but we build hist 
      // for it to perform hist subtraction, remove here
      hist_manager->remove(nid);
    }
  }
  TOK(find);
  HML_LOG_DEBUG << "Find splits for nodes " << nids 
    << " cost " << COST_MSEC(find) << " ms";
  return splits;
}

std::map<uint32_t, uint32_t> 
GBDTTrainer::DoSplitNodes(const GBTSplitPtrMap& splits, int class_id) {
  std::map<uint32_t, uint32_t> chilren_sizes;
  for (auto const& t : splits) {
    nid_t nid = t.first;
    auto& split = t.second;
    auto& split_entry = split->get_split_entry();
    node_indexer->UpdatePos(nid, *train_data, split_entry, 
      feat_info->splits[split_entry.get_fid()]);
    node_indexer->set_node_gp(2 * nid + 1, split->get_left_gp());
    node_indexer->set_node_gp(2 * nid + 2, split->get_right_gp());
    chilren_sizes[2 * nid + 1] = node_indexer->get_node_size(2 * nid + 1);
    chilren_sizes[2 * nid + 2] = node_indexer->get_node_size(2 * nid + 2);
  }
  return chilren_sizes;
}

void GBDTTrainer::DoSetAsLeaves(const std::vector<nid_t>& nids) {
  for (nid_t nid : nids) {
    hist_manager->remove(nid);
    if (nid != 0) hist_manager->remove(PARENT(nid));
  }
}

void GBDTTrainer::DoUpdatePreds(GBTree& tree, float lr, int class_id) {
  if (!param->IsMultiClass()) {
    // regression or binary-classification tasks
    for (auto const& node : tree.get_nodes()) {
      if (node != nullptr && node->is_leaf()) {
        if (!param->is_majority_voting) {
          node_indexer->UpdatePreds(
            node->get_nid(), ins_info->predictions, 
            node->get_weight(), lr);
        } else {
          float vote_pred = node->get_weight() >= 0 ? 1 : -1;
          node_indexer->UpdatePreds(
            node->get_nid(), ins_info->predictions, 
            vote_pred, 1.0);
        }
      }
    }
  } else if (!param->IsMultiClassMultiTree()) {
    // multi-classification tasks, use multi-class trees
    for (auto const& node : tree.get_nodes()) {
      if (node != nullptr && node->is_leaf()) {
        if (!param->is_majority_voting) {
          node_indexer->UpdatePreds(
            node->get_nid(), ins_info->predictions, 
            node->get_weights(), lr);
        } else {
          std::vector<float> vote_preds(param->num_label, 0);
          auto vote_class_id = Argmax(node->get_weights().data(), 
            param->num_label);
          vote_preds[vote_class_id] = 1;
          node_indexer->UpdatePreds(
            node->get_nid(), ins_info->predictions, 
            vote_preds, 1.0);
        }
      }
    }
  } else {
    // multi-classification tasks, use multiple one-vs-rest trees
    ASSERT(!param->is_majority_voting) 
      << "Majority voting with multiple one-vs-rest trees is not supported";
    for (auto const& node : tree.get_nodes()) {
      if (node != nullptr && node->is_leaf()) {
        node_indexer->UpdatePreds(
          node->get_nid(), ins_info->predictions, 
          node->get_weight(), lr, 
          param->num_label, class_id);
      }
    }
  }

  if (num_valid > 0) {
    bool is_dense = train_data->is_dense();
    int num_classes = param->num_label;
    if (!param->IsMultiClass()) {
      // regression or binary-classification tasks
      #pragma omp parallel for
      for (uint32_t i = 0; i < valid_data->get_num_instances(); i++) {
        float weight = is_dense 
          ? tree.predict_scalar(valid_data->get_dense_feature(i))
          : tree.predict_scalar(valid_data->get_sparse_feature(i));
        if (!param->is_majority_voting) {
          valid_preds[i] += weight * lr;
        } else {
          valid_preds[i] += weight >= 0 ? 1 : -1;
        }
      }
    } else if (!param->IsMultiClassMultiTree()) {
      // multi-classification tasks, use multi-class trees
      #pragma omp parallel for
      for (uint32_t i = 0; i < valid_data->get_num_instances(); i++) {
        const auto& weights = is_dense 
          ? tree.predict_vector(valid_data->get_dense_feature(i))
          : tree.predict_vector(valid_data->get_sparse_feature(i));
        uint32_t offset = i * num_classes;
        if (!param->is_majority_voting) {
          for (int k = 0; k < num_classes; k++)
            valid_preds[offset + k] += weights[k] * lr;
        } else {
          auto vote_class_id = Argmax(weights.data(), num_classes);
          valid_preds[offset + vote_class_id] += 1;
        }
      }
    } else {
      // multi-classification tasks, use multiple one-vs-rest trees
      #pragma omp parallel for
      for (uint32_t i = 0; i < valid_data->get_num_instances(); i++) {
        float weight = is_dense 
          ? tree.predict_scalar(valid_data->get_dense_feature(i))
          : tree.predict_scalar(valid_data->get_sparse_feature(i));
        valid_preds[i * num_classes + class_id] += weight * lr;
      }
    }
  }
}

std::map<std::string, std::tuple<double, double>> 
GBDTTrainer::DoEvaluate(int round_id) {
  if (param->is_majority_voting) {
    auto& train_preds = ins_info->predictions;
    #pragma omp parallel for
    for (uint32_t i = 0; i < train_preds.size(); i++)
      train_preds[i] /= round_id + 1;
    if (num_valid > 0) {
      #pragma omp parallel for
      for (uint32_t i = 0; i < valid_preds.size(); i++)
        valid_preds[i] /= round_id + 1;
    }
  }

  std::map<std::string, std::tuple<double, double>> metrics;
  for (auto const& metric_name : this->param->metrics) {
    const auto* metric = \
      MetricFactory::GetEvalMetric<float, float, double>(metric_name);
    double train = 0, valid = 0;
    if (!param->IsMultiClass()) {
      train = metric->EvalBinary(ins_info->predictions.data(), 
        train_data->get_labels().data(), num_train);
      if (num_valid > 0) {
        valid = metric->EvalBinary(valid_preds.data(), 
          valid_data->get_labels().data(), num_valid);
      }
    } else {
      train = metric->EvalMulti(ins_info->predictions.data(), 
        train_data->get_labels().data(), param->num_label, num_train);
      if (num_valid > 0) {
        valid = metric->EvalMulti(valid_preds.data(), 
          valid_data->get_labels().data(), param->num_label, num_valid);
      }
    }
    metrics[metric_name] = std::make_tuple(train, valid);
  }

  if (param->is_majority_voting) {
    auto& train_preds = ins_info->predictions;
    #pragma omp parallel for
    for (uint32_t i = 0; i < train_preds.size(); i++)
      train_preds[i] *= round_id + 1;
    if (num_valid > 0) {
      #pragma omp parallel for
      for (uint32_t i = 0; i < valid_preds.size(); i++)
        valid_preds[i] *= round_id + 1;
    }
  }

  return metrics;
}

} // namespace gbdt
} // namespace ml
} // namespace hetu
