#ifndef __HETU_ML_MODEL_TREE_TREE_H_
#define __HETU_ML_MODEL_TREE_TREE_H_

#include "model/common/mlbase.h"
#include "model/tree/split.h"

namespace hetu { 
namespace ml {
namespace tree {

#define MAX_NODE_NUM(depth) (Pow(2, (depth) + 1) - 1)
#define MAX_INNER_NODE_NUM(depth) (Pow(2, (depth)) - 1)
#define LAYER_NODE_NUM(depth) (Pow(2, (depth)))
#define PARENT(nid) (((nid) - 1) / 2)
#define SIBLING(nid) (IS_EVEN((nid)) ? ((nid) - 1) : ((nid) + 1))

typedef uint32_t nid_t;

namespace TreeConf {
  // max depth
  static const std::string MAX_DEPTH = "MAX_DEPTH";
  static const int DEFAULT_MAX_DEPTH = 4;
  // max node num
  static const std::string MAX_NODE_NUM = "MAX_NODE_NUM";
  static const int DEFAULT_MAX_NODE_NUM = MAX_NODE_NUM(DEFAULT_MAX_DEPTH);
  // leaf-wise or not
  static const std::string LEAFWISE = "LEAFWISE";
  static const bool DEFAULT_LEAFWISE = false;
  // number of candidate splits per feature
  static const std::string NUM_SPLIT = "NUM_SPLIT";
  static const int DEFAULT_NUM_SPLIT = 10;
  // instance sampling ratio
  static const std::string INS_SP_RATIO = "INS_SP_RATIO";
  static const float DEFAULT_INS_SP_RATIO = 1;
  // feature sampling ratio
  static const std::string FEAT_SP_RATIO = "FEAT_SP_RATIO";
  static const float DEFAULT_FEAT_SP_RATIO = 1;

  static std::vector<std::string> meaningful_keys() {
    return {
      MAX_DEPTH, 
      MAX_NODE_NUM, 
      LEAFWISE, 
      NUM_SPLIT, 
      INS_SP_RATIO, 
      FEAT_SP_RATIO
    };
  }

  static Args default_args() { 
    return {
      { MAX_DEPTH, std::to_string(DEFAULT_MAX_DEPTH) }, 
      { MAX_NODE_NUM, std::to_string(DEFAULT_MAX_NODE_NUM) }, 
      { LEAFWISE, std::to_string(DEFAULT_LEAFWISE) }, 
      { NUM_SPLIT, std::to_string(DEFAULT_NUM_SPLIT) }, 
      { INS_SP_RATIO, std::to_string(DEFAULT_INS_SP_RATIO) }, 
      { FEAT_SP_RATIO, std::to_string(DEFAULT_FEAT_SP_RATIO) }
    };
  }
} // TreeConf

class TreeParam : public MLParam {
public:
  TreeParam(const Args& args = {}, 
            const Args& default_args = TreeConf::default_args())
  : MLParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return TreeConf::meaningful_keys();
  }

  int max_depth;
  int max_node_num;
  bool leafwise;
  int num_split;
  float ins_sp_ratio;
  float feat_sp_ratio;

protected:
  inline void InitAndCheckParam() override {
    this->max_depth = argparse::Get<int>(
      this->all_args, TreeConf::MAX_DEPTH);
    ASSERT_GT(this->max_depth, 0) 
      << "Invalid max depth: " << this->max_depth;
    this->max_node_num = MIN(argparse::Get<int>(
      this->all_args, TreeConf::MAX_NODE_NUM), 
      MAX_NODE_NUM(this->max_depth));
    ASSERT_GT(this->max_node_num, 0) 
      << "Invalid max number of nodes: " << this->max_node_num;
    this->leafwise = argparse::GetBool(
      this->all_args, TreeConf::LEAFWISE);
    this->num_split = argparse::Get<int>(
      this->all_args, TreeConf::NUM_SPLIT);
    ASSERT(this->num_split > 0 && this->num_split < 256)
      << "Invalid number of candidate splits: " << this->num_split;
    this->ins_sp_ratio = argparse::Get<float>(
      this->all_args, TreeConf::INS_SP_RATIO);
    ASSERT(this->ins_sp_ratio > 0 & this->ins_sp_ratio <= 1)
      << "Invalid instance sampling ratio: " << this->ins_sp_ratio;
    this->feat_sp_ratio = argparse::Get<float>(
      this->all_args, TreeConf::FEAT_SP_RATIO);
    ASSERT(this->feat_sp_ratio > 0 & this->feat_sp_ratio <= 1)
      << "Invalid feature sampling ratio: " << this->feat_sp_ratio;
  }
};

class TNode {
public:
  TNode(const nid_t nid): nid(nid), split_entry(nullptr), leaf(false) {}

  ~TNode() {
    if (this->split_entry != nullptr) 
      delete this->split_entry;
  }

  inline nid_t get_nid() const { return nid; }

  inline const SplitEntry& get_split_entry() const { 
    return *split_entry; 
  }

  inline void set_split_entry(const SplitEntry& split_entry) {
    if (this->split_entry != nullptr) 
      delete this->split_entry;
    this->split_entry = split_entry.copy();
  }

  inline bool is_leaf() const { return leaf; }

  inline void chg_to_leaf() { leaf = true; }

protected:
  const nid_t nid;
  SplitEntry* split_entry;
  bool leaf;
};

template <typename Node> 
struct is_node { 
  static const bool value = std::is_base_of<TNode, Node>::value;
};

template <typename Node>
class Tree {
public:
  Tree(int max_depth): nodes(MAX_NODE_NUM(max_depth)), num_nodes(0) {
    static_assert(is_node<Node>::value, 
      "The template class is not derived from TNode");
  }

  inline const Node& get_root() const { return get_node(0); }

  inline void set_root(std::unique_ptr<Node>& root) { set_node(0, root); }

  inline const Node& get_node(nid_t nid) const { return *(this->nodes[nid]); }

  inline void set_node(nid_t nid, std::unique_ptr<Node>& node) {
    ASSERT(nid == node->get_nid()) 
      << "Node id mismatch: " << nid << " vs. " << node->get_nid();
    ASSERT(nid < this->nodes.size()) 
      << "Node id " << nid << " exceeds maximum node size";
    ASSERT(nodes[nid] == nullptr) 
      << "Setting node[" << nid << "] twice is not allowed";
    ASSERT(node != nullptr) 
      << "Trying to set a nullptr for node[" << nid << "]";
    this->nodes[nid].swap(node);
    this->num_nodes++;
  }

  inline const std::vector<std::unique_ptr<Node>>& get_nodes() const { 
    return this->nodes; 
  }

  inline int get_num_nodes() const { return this->num_nodes; }

  inline int get_num_leaves() const { return (this->num_nodes + 1) / 2; }

protected:
  std::vector<std::unique_ptr<Node>> nodes;
  int num_nodes;
};

} // namespace tree
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_TREE_TREE_H_
