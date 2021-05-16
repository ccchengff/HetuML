#ifndef __HETU_ML_MODEL_GBDT_MODEL_H_
#define __HETU_ML_MODEL_GBDT_MODEL_H_

#include "model/common/mlbase.h"
#include "model/tree/tree.h"
#include "model/gbdt/hist/grad_pair.h"

namespace hetu { 
namespace ml {
namespace gbdt {

using namespace hetu::ml::tree;
using hetu::ml::operator<<;

namespace GBDTConf {
  // is regression or not
  static const std::string IS_REGRESSION = "IS_REGRESSION";
  static const bool DEFAULT_IS_REGRESSION = false;
  // number of classes
  static const std::string NUM_LABEL = "NUM_LABEL";
  static const int DEFAULT_NUM_LABEL = 2;
  // number of training rounds
  static const std::string NUM_ROUND = "NUM_ROUND";
  static const int DEFAULT_NUM_ROUND = 10;
  // learning rate
  static const std::string LEARNING_RATE = "LEARNING_RATE";
  static const float DEFAULT_LEARNING_RATE = 0.1;
  // minimal split gain
  static const std::string MIN_SPLIT_GAIN = "MIN_SPLIT_GAIN";
  static const float DEFAULT_MIN_SPLIT_GAIN = 0;
  // minimal number of instances in node
  static const std::string MIN_NODE_INS = "MIN_NODE_INS";
  static const int DEFAULT_MIN_NODE_INS = 1024;
  // minimal hessian weight of node
  static const std::string MIN_CHILD_WEIGHT = "MIN_CHILD_WEIGHT";
  static const float DEFAULT_MIN_CHILD_WEIGHT = 0;
  // l1 reg term
  static const std::string REG_ALPHA = "REG_ALPHA";
  static const float DEFAULT_REG_ALPHA = 0;
  // l2 reg term
  static const std::string REG_LAMBDA = "REG_LAMBDA";
  static const float DEFAULT_REG_LAMBDA = 1;
  // maximal prediction weight of tree leaf
  static const std::string MAX_LEAF_WEIGHT = "MAX_LEAF_WEIGHT";
  static const float DEFAULT_MAX_LEAF_WEIGHT = 0;
  // use multiple one-vs-rest trees, only valid in multi-class tasks
  static const std::string MULTI_TREE = "MULTI_TREE";
  static const bool DEFAULT_MULTI_TREE = false;
  // whether to use majority 
  static const std::string IS_MAJORITY_VOTING = "IS_MAJORITY_VOTING";
  static const bool DEFAULT_IS_MAJORITY_VOTING = false;
  // loss function and evaluation metrics
  // loss functions: logistic, rmse
  // evaluation metrics: log-loss, cross-entropy, error, precision, rmse
  static const std::string LOSS = "LOSS";
  static const std::string METRICS = "METRICS";

  static std::vector<std::string> meaningful_keys() {
    const auto& tree_keys = TreeConf::meaningful_keys();
    std::vector<std::string> gbdt_keys = {
      IS_REGRESSION, 
      NUM_LABEL, 
      NUM_ROUND, 
      LEARNING_RATE, 
      MIN_SPLIT_GAIN, 
      MIN_NODE_INS, 
      MIN_CHILD_WEIGHT, 
      REG_ALPHA, 
      REG_LAMBDA, 
      MAX_LEAF_WEIGHT, 
      MULTI_TREE, 
      IS_MAJORITY_VOTING, 
      LOSS, 
      METRICS
    };
    gbdt_keys.insert(gbdt_keys.end(), tree_keys.begin(), tree_keys.end());
    return std::move(gbdt_keys);
  }

  static Args default_args() { 
    const auto& tree_args = TreeConf::default_args();
    Args gbdt_args = {
      { IS_REGRESSION, std::to_string(DEFAULT_IS_REGRESSION) }, 
      { NUM_LABEL, std::to_string(DEFAULT_NUM_LABEL) }, 
      { NUM_ROUND, std::to_string(DEFAULT_NUM_ROUND) }, 
      { LEARNING_RATE, std::to_string(DEFAULT_LEARNING_RATE) }, 
      { MIN_SPLIT_GAIN, std::to_string(DEFAULT_MIN_SPLIT_GAIN) }, 
      { MIN_NODE_INS, std::to_string(DEFAULT_MIN_NODE_INS) }, 
      { MIN_CHILD_WEIGHT, std::to_string(DEFAULT_MIN_CHILD_WEIGHT) }, 
      { REG_ALPHA, std::to_string(DEFAULT_REG_ALPHA) }, 
      { REG_LAMBDA, std::to_string(DEFAULT_REG_LAMBDA) }, 
      { MAX_LEAF_WEIGHT, std::to_string(DEFAULT_MAX_LEAF_WEIGHT) }, 
      { MULTI_TREE, std::to_string(DEFAULT_MULTI_TREE) }, 
      { IS_MAJORITY_VOTING, std::to_string(DEFAULT_IS_MAJORITY_VOTING) }, 
      { LOSS, "auto" }, 
      { METRICS, "" }
    };
    gbdt_args.insert(tree_args.begin(), tree_args.end());
    return std::move(gbdt_args);
  }
} // GBDTConf

class GBDTParam : public TreeParam {
public:
  GBDTParam(const Args& args = {}, 
            const Args& default_args = GBDTConf::default_args())
  : TreeParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return GBDTConf::meaningful_keys();
  }

  inline bool IsMultiClass() const {
    return !is_regression && num_label > 2;
  }

  inline bool IsLeafVector() const {
    return IsMultiClass() && !multi_tree;
  }
  
  inline bool IsMultiClassMultiTree() const {
    return IsMultiClass() && multi_tree;
  }

  inline uint32_t pred_size() const {
    return IsMultiClass() ? num_label : 1;
  }

  inline uint32_t m1_size() const { return pred_size(); }

  inline uint32_t m2_size() const {
    return IsMultiClass() ? num_label : 1; // diagonal
    // return IsMultiClass() \
    //   ? num_label * (num_label + 1) / 2 : 1; // full hessian
  }

  bool is_regression;
  int num_label;
  int num_round;
  float learning_rate;
  float min_split_gain;
  int min_node_ins;
  float min_child_weight;
  float reg_alpha;
  float reg_lambda;
  float max_leaf_weight;
  bool multi_tree;
  bool is_majority_voting;
  std::string loss;
  std::vector<std::string> metrics;
private:
  inline void InitAndCheckParam() override {
    TreeParam::InitAndCheckParam();
    this->is_regression = argparse::GetBool(
      this->all_args, GBDTConf::IS_REGRESSION);
    this->num_label = argparse::Get<int>(
      this->all_args, GBDTConf::NUM_LABEL);
    ASSERT_GT(this->num_label, 0) 
      << "Invalid number of labels: " << this->num_label;
    this->num_round = argparse::Get<int>(
      this->all_args, GBDTConf::NUM_ROUND);
    ASSERT_GT(this->num_round, 0) 
      << "Invalid number of rounds: " << this->num_round;
    this->learning_rate = argparse::Get<float>(
      this->all_args, GBDTConf::LEARNING_RATE);
    this->min_split_gain = argparse::Get<float>(
      this->all_args, GBDTConf::MIN_SPLIT_GAIN);
    this->min_node_ins = argparse::Get<float>(
      this->all_args, GBDTConf::MIN_NODE_INS);
    this->min_child_weight = argparse::Get<float>(
      this->all_args, GBDTConf::MIN_CHILD_WEIGHT);
    this->reg_alpha = argparse::Get<float>(
      this->all_args, GBDTConf::REG_ALPHA);
    this->reg_lambda = argparse::Get<float>(
      this->all_args, GBDTConf::REG_LAMBDA);
    this->max_leaf_weight = argparse::Get<float>(
      this->all_args, GBDTConf::MAX_LEAF_WEIGHT);
    this->multi_tree = argparse::GetBool(
      this->all_args, GBDTConf::MULTI_TREE);
    this->is_majority_voting = argparse::GetBool(
      this->all_args, GBDTConf::IS_MAJORITY_VOTING);
    if (this->is_regression) {
      this->loss = RichSquareLoss<label_t, label_t, label_t>::NAME;
    } else {
      this->loss = RichLogisticLoss<label_t, label_t, label_t>::NAME;
    }
    this->metrics = argparse::GetVector<std::string>(
      this->all_args, GBDTConf::METRICS);
    // check whether metrics are properly set
    for (const auto& metric : this->metrics) {
      const auto* eval_metric = \
        MetricFactory::GetEvalMetric<label_t, label_t, label_t>(metric, false);
      ASSERT(eval_metric != nullptr) << "Undefined metric: " << metric;
    }
  }
};

class GBTSplit {
public:

  GBTSplit(SplitEntry* split_entry, GradPair* left_gp, GradPair* right_gp)
  : split_entry(split_entry), left_gp(left_gp), right_gp(right_gp) {}

  GBTSplit(std::unique_ptr<SplitEntry>& split_entry, 
           std::unique_ptr<GradPair>& left_gp, 
           std::unique_ptr<GradPair>& right_gp) {
    this->split_entry.swap(split_entry);
    this->left_gp.swap(left_gp);
    this->right_gp.swap(right_gp);
  }

  inline bool is_valid(float min_gain) const {
    return split_entry != nullptr && !split_entry->is_empty() && 
      split_entry->get_gain() > min_gain;
  }

  inline bool NeedReplace(const GBTSplit& other) const {
    return this->split_entry->NeedReplace(*other.split_entry);
  }

  inline bool Update(const GBTSplit& other) {
    return Update(*other.split_entry, *other.left_gp, *other.right_gp);
  }

  inline bool Update(const SplitEntry& split_entry, 
                     const GradPair& left_gp, 
                     const GradPair& right_gp) {
    if (this->split_entry->NeedReplace(split_entry)) {
      this->split_entry.reset(split_entry.copy());
      this->left_gp->set(left_gp);
      this->right_gp->set(right_gp);
      return true;
    }
    return false;
  }

  inline const SplitEntry& get_split_entry() const { return *split_entry; }

  inline const GradPair& get_left_gp() const { return *left_gp; }

  inline const GradPair& get_right_gp() const { return *right_gp; }

  friend std::ostream& operator<<(std::ostream& os, GBTSplit& split) {
    os << "{ split_entry: " << *split.split_entry 
      << ", left_gp: " << *split.left_gp
      << ", right_gp: " << *split.right_gp << " }";
    return os;
  }

private:
  std::unique_ptr<SplitEntry> split_entry;
  std::unique_ptr<GradPair> left_gp;
  std::unique_ptr<GradPair> right_gp;
};

class GBTNode : public TNode {
public:
  GBTNode(const nid_t nid): TNode(nid), size(0), gain(0) {}

  inline uint32_t get_size() const { return size; }

  inline void set_size(uint32_t size) { this->size = size; }

  inline float get_gain() const { return gain; }

  inline void set_gain(float gain) { this->gain = gain; }

  inline const GradPair& get_sum_gp() const { return *sum_gp; }

  inline void set_sum_gp(const GradPair& sum_gp) { 
    if (this->sum_gp == nullptr)
      this->sum_gp.reset(sum_gp.copy());
    else
      this->sum_gp->set(sum_gp);
  }

  inline float get_weight() const { return weights[0]; }

  inline void set_weight(float weight) {
    this->weights.resize(1);
    this->weights[0] = weight;
  }

  inline const std::vector<float>& get_weights() const { return weights; }

  inline void set_weights(const std::vector<float>& weights) {
    this->weights.resize(weights.size());
    std::copy(weights.begin(), weights.end(), this->weights.begin());
  }

  inline void set_weights(const std::vector<double>& weights) {
    this->weights.resize(weights.size());
    std::copy(weights.begin(), weights.end(), this->weights.begin());
  }

private:
  uint32_t size;
  float gain;
  std::unique_ptr<GradPair> sum_gp;
  std::vector<float> weights;
};

class GBTree : public Tree<GBTNode> {
public:
  GBTree(int max_depth): Tree<GBTNode>(max_depth) {}

  template<typename T>
  inline const float predict_scalar(const DenseVector<T>& x) const {
    return flow_to_leaf(x).get_weight();
  }

  template<typename T>
  inline const float predict_scalar(const SparseVector<T>& x) const {
    return flow_to_leaf(x).get_weight();
  }

  template<typename T>
  inline const std::vector<float>& 
  predict_vector(const DenseVector<T>& x) const {
    return flow_to_leaf(x).get_weights();
  }

  template<typename T>
  inline const std::vector<float>& 
  predict_vector(const SparseVector<T>& x) const {
    return flow_to_leaf(x).get_weights();
  }

  template <typename T>
  inline const GBTNode& flow_to_leaf(const DenseVector<T>& x) const { 
    nid_t nid = 0;
    const auto& values = x.values;
    while (true) {
      if (nodes[nid]->is_leaf()) {
        return *nodes[nid];
      } else {
        const auto& split_entry = nodes[nid]->get_split_entry();
        int fid = split_entry.get_fid();
        int flow_to = std::isnan(values[fid]) 
          ? split_entry.FlowTo(values[fid])
          : split_entry.DefaultTo();
        nid = 2 * nid + 1 + flow_to;
      }
    }
  }

  template <typename T>
  inline const GBTNode& flow_to_leaf(const SparseVector<T>& x) const { 
    nid_t nid = 0;
    const auto& indices = x.indices;
    const auto& values = x.values;
    const uint32_t nnz = x.nnz;
    while (true) {
      if (nodes[nid]->is_leaf()) {
        return *nodes[nid];
      } else {
        const auto& split_entry = nodes[nid]->get_split_entry();
        int fid = split_entry.get_fid();
        auto p = std::lower_bound(indices, indices + nnz, fid);
        int flow_to = (p - indices < nnz && *p == fid) 
          ? split_entry.FlowTo(values[p - indices]) 
          : split_entry.DefaultTo();
        nid = 2 * nid + 1 + flow_to;
      }
    }
  }
};

class GBDTModel {
public:
  GBDTModel(): GBDTModel(new GBDTParam()) {}

  GBDTModel(const GBDTParam& param): GBDTModel(new GBDTParam(param)) {}

  GBDTModel(GBDTParam* param): param(param) {}

  GBDTModel(std::shared_ptr<GBDTParam> param): param(param) {}

  template<typename T>
  inline const float predict_scalar(const AVector<T>* x) const {
    if (x->is_dense())
      return predict_scalar((const DenseVector<T>&) *x);
    else
      return predict_scalar((const SparseVector<T>&) *x);
  }

  template<typename T>
  inline const float predict_scalar(const DenseVector<T>& x) const {
    if (!param->is_majority_voting) {
      float lr = param->learning_rate;
      float pred = init_preds[0];
      for (const auto& tree : trees) {
        pred += lr * tree->predict_scalar(x);
      }
      return pred;
    } else {
      int counts = 0;
      for (const auto& tree : trees) {
        counts += tree->predict_scalar(x) > 0 ? 1 : -1;
      }
      return 1.0 * counts / num_trees();
    }
  }

  template<typename T>
  inline const float predict_scalar(const SparseVector<T>& x) const {
    if (!param->is_majority_voting) {
      float lr = param->learning_rate;
      float pred = init_preds[0];
      for (const auto& tree : trees) {
        pred += lr * tree->predict_scalar(x);
      }
      return pred;
    } else {
      int counts = 0;
      for (const auto& tree : trees) {
        counts += tree->predict_scalar(x) > 0 ? 1 : -1;
      }
      return 1.0 * counts / num_trees();
    }
  }

  template<typename T>
  inline void predict_vector(const AVector<T>* x, 
                             float* const preds) const {
    if (x->is_dense())
      predict_vector((const DenseVector<T>&) *x, preds);
    else
      predict_vector((const SparseVector<T>&) *x, preds);
  }

  template<typename T>
  inline void predict_vector(const DenseVector<T>& x, 
                             float* const preds) const {
    int num_classes = param->num_label;
    if (!param->is_majority_voting) {
      float lr = param->learning_rate;
      std::copy(init_preds.begin(), init_preds.end(), preds);
      if (!param->IsMultiClassMultiTree()) {
        for (const auto& tree : trees) {
          const auto& scores = tree->predict_vector(x);
          for (int k = 0; k < num_classes; k++) {
            preds[k] += lr * scores[k];
          }
        }
      } else {
        for (int tree_id = 0; tree_id < trees.size(); tree_id++) {
          auto score = trees[tree_id]->predict_scalar(x);
          preds[tree_id % num_classes] += lr * score;
        }
      }
    } else {
      for (const auto& tree : trees) {
        const auto& scores = tree->predict_vector(x);
        auto class_id = Argmax(scores.data(), num_classes);
        preds[class_id] += 1;
      }
      for (int k = 0; k < num_classes; k++) {
        preds[k] /= num_trees();
      }
    }
  }

  template<typename T>
  inline void predict_vector(const SparseVector<T>& x, 
                             float* const preds) const {
    int num_classes = param->num_label;
    if (!param->is_majority_voting) {
      float lr = param->learning_rate;
      std::copy(init_preds.begin(), init_preds.end(), preds);
      if (!param->IsMultiClassMultiTree()) {
        for (const auto& tree : trees) {
          const auto& scores = tree->predict_vector(x);
          for (int k = 0; k < num_classes; k++) {
            preds[k] += lr * scores[k];
          }
        }
      } else {
        for (int tree_id = 0; tree_id < trees.size(); tree_id++) {
          auto score = trees[tree_id]->predict_scalar(x);
          preds[tree_id % num_classes] += lr * score;
        }
      }
    } else {
      for (const auto& tree : trees) {
        const auto& scores = tree->predict_vector(x);
        auto class_id = Argmax(scores.data(), num_classes);
        preds[class_id] += 1;
      }
      for (int k = 0; k < num_classes; k++) {
        preds[k] /= num_trees();
      }
    }
  }

  inline const GBDTParam& get_param() const { return *param; }

  inline const std::vector<float>& get_init_preds() const { 
    return init_preds; 
  }

  inline void set_init_preds(const std::vector<float>& init_preds) {
    this->init_preds.resize(init_preds.size());
    std::copy(init_preds.begin(), init_preds.end(), this->init_preds.begin());
  }

  inline const GBTree& get(int index) const { return *trees[index]; }

  inline void add(std::unique_ptr<GBTree>& tree) { 
    trees.push_back(std::move(tree)); 
  }

  inline const GBTree& head() const { return get(0); }

  inline const GBTree& last() const { return get(num_trees() - 1); }

  inline int num_trees() const { return trees.size(); }

  std::string ToString(bool save_stats=false) const {
    std::ostringstream os;
    const char indent = '\t';
    const char delim = '\n';
    os << *param << delim;
    os << "init_preds=" << init_preds << delim;
    os << "num_trees=" << trees.size() << delim;
    os << "save_stats=" << save_stats << delim;
    for (int tree_id = 0; tree_id < trees.size(); tree_id++) {
      const auto& tree = trees.at(tree_id);
      if (tree != nullptr) {
        os << indent << "tree_id=" << tree_id << delim;
        os << indent << "num_nodes=" << tree->get_num_nodes() << delim;
        for (const auto& node : tree->get_nodes()) {
          if (node != nullptr) {
            os << indent << indent << "node_id=" << node->get_nid() << delim;
            os << indent << indent << "is_leaf=" << node->is_leaf() << delim;
            if (!node->is_leaf()) {
              os << indent << indent << "split_entry=" << node->get_split_entry() << delim;
            } else {
              if (!param->IsLeafVector()) {
                os << indent << indent << "weights=" << node->get_weight() << delim;
              } else {
                os << indent << indent << "weights=" << node->get_weights() << delim;
              }
            }
            if (save_stats) {
              os << indent << indent << "size=" << node->get_size() << delim;
              os << indent << indent << "gain=" << node->get_gain() << delim;
              os << indent << indent << "sum_gp=" << node->get_sum_gp() << delim;
            }
          }
        }
      }
    }
    return os.str();
  }

  void FromString(const std::string& str) {
    std::istringstream is(str);
    is >> *param;

    std::string buf;
    size_t offset;
    std::getline(is, buf); // empty line
    // init preds
    std::getline(is, buf);
    offset = argparse::GetOffset(buf, "init_preds=");
    argparse::ParseVector<float>(buf.substr(offset), init_preds);
    // num trees
    std::getline(is, buf);
    offset = argparse::GetOffset(buf, "num_trees=");
    int num_trees = argparse::Parse<int>(buf.substr(offset));
    this->trees.clear(); this->trees.resize(num_trees);
    // wheter statistics are saved
    std::getline(is, buf);
    offset = argparse::GetOffset(buf, "save_stats=");
    bool save_stats = argparse::ParseBool(buf.substr(offset));
    for (int i = 0; i < num_trees; i++) {
      std::getline(is, buf);
      offset = argparse::GetOffset(buf, "tree_id=");
      int tree_id = argparse::Parse<int>(buf.substr(offset));
      std::getline(is, buf);
      offset = argparse::GetOffset(buf, "num_nodes=");
      int num_nodes = argparse::Parse<int>(buf.substr(offset));
      std::unique_ptr<GBTree> tree(new GBTree(param->max_depth));
      for (int j = 0; j < num_nodes; j++) {
        // node id
        std::getline(is, buf);
        offset = argparse::GetOffset(buf, "node_id=");
        int node_id = argparse::Parse<int>(buf.substr(offset));
        std::unique_ptr<GBTNode> node(new GBTNode(node_id));
        // is leaf
        std::getline(is, buf);
        offset = argparse::GetOffset(buf, "is_leaf=");
        bool is_leaf = argparse::ParseBool(buf.substr(offset));
        if (is_leaf) {
          node->chg_to_leaf();
          // leaf weight
          std::getline(is, buf);
          offset = argparse::GetOffset(buf, "weights=");
          if (!param->IsLeafVector()) {
            node->set_weight(argparse::Parse<float>(buf.substr(offset)));
          } else {
            std::vector<float> weights;
            argparse::ParseVector<float>(buf.substr(offset), weights);
            node->set_weights(weights);
          }
        } else {
          // split entry
          std::getline(is, buf);
          offset = argparse::GetOffset(buf, "split_entry=");
          SplitPoint sp;
          sp.FromString(buf.substr(offset));
          node->set_split_entry(sp);
        }
        if (save_stats) {
          // node size
          std::getline(is, buf);
          offset = argparse::GetOffset(buf, "size=");
          node->set_size(argparse::Parse<int>(buf.substr(offset)));
          // node gain
          std::getline(is, buf);
          offset = argparse::GetOffset(buf, "gain=");
          node->set_gain(argparse::Parse<float>(buf.substr(offset)));
          // sum grad pair
          std::getline(is, buf);
          offset = argparse::GetOffset(buf, "sum_gp=");
          if (!param->IsLeafVector()) {
            BinaryGradPair binary;
            binary.FromString(buf.substr(offset));
            node->set_sum_gp(binary);
          } else {
            MultiGradPair multi(param->num_label);
            multi.FromString(buf.substr(offset));
            node->set_sum_gp(multi);
          }
        }
        tree->set_node(node_id, node);
      }
      trees[tree_id] = std::move(tree);
    }
  }

private:
  std::shared_ptr<GBDTParam> param;
  std::vector<std::unique_ptr<GBTree>> trees;
  std::vector<float> init_preds;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_MODEL_H_
