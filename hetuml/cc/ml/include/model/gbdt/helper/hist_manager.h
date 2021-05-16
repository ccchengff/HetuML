#ifndef __HETU_ML_MODEL_GBDT_HELPER_HIST_MANAGER_H_
#define __HETU_ML_MODEL_GBDT_HELPER_HIST_MANAGER_H_

#include "common/logging.h"
#include "common/threading.h"
#include "data/dataset.h"
#include "model/gbdt/model.h"
#include "model/gbdt/hist/histogram.h"
#include "model/gbdt/helper/metadata.h"
#include "model/gbdt/helper/node_indexer.h"
#include <memory>

namespace hetu { 
namespace ml {
namespace gbdt {

/******************************************************
 * Builder & Storage for histograms
 ******************************************************/
class HistManager {
public:

  HistManager(int max_depth, int num_classes, bool multi_tree)
  : num_classes(num_classes), multi_tree(multi_tree) {
    int max_node_num = MAX_NODE_NUM(max_depth);
    node_hists.resize(max_node_num, nullptr);
    hist_store.resize(max_node_num + OMP_GET_NUM_THREADS() * 2, nullptr);
    avail_hist = 0;
  }

  void Reset() {
    std::fill(node_hists.begin(), node_hists.end(), nullptr);
    std::fill(hist_store.begin(), hist_store.end(), nullptr);
    avail_hist = 0;
  }

  void BuildHistForRoot(const CompactDataset<float, int>& dataset, 
                        const InstanceInfo& ins_info, 
                        const FeatureInfo& feat_info, 
                        const NodeIndexer& node_indexer, 
                        int class_id = -1) {
    std::shared_ptr<NodeHist> node_hist = nullptr;
    if (node_indexer.get_node_size(0) == dataset.get_num_instances()) {
      node_hist = SparseBuild0(dataset, ins_info, feat_info, 
        node_indexer, class_id);
    } else {
      node_hist = SparseBuild1(0, dataset, ins_info, feat_info, 
        node_indexer, class_id);
    }
    if (!dataset.is_dense())
      FillDefaultBins(feat_info, node_indexer.get_node_gp(0), *node_hist);
    node_hists[0] = node_hist;
  }

  void BuildHistForNodes(const std::vector<nid_t>& nids, 
                         const CompactDataset<float, int>& dataset, 
                         const InstanceInfo& ins_info, 
                         const FeatureInfo& feat_info, 
                         const NodeIndexer& node_indexer, 
                         int class_id = -1) {
    for (nid_t nid : nids) {
      std::shared_ptr<NodeHist> node_hist = SparseBuild1(nid, 
        dataset, ins_info, feat_info, node_indexer, class_id);
      if (!dataset.is_dense())
        FillDefaultBins(feat_info, node_indexer.get_node_gp(nid), *node_hist);
      node_hists[nid] = node_hist;
    }
  }

  void HistSubtract(nid_t nid) {
    nid_t par_nid = PARENT(nid);
    nid_t sib_nid = SIBLING(nid);
    NodeHist& par_hist = *node_hists[par_nid];
    NodeHist& sib_hist = *node_hists[sib_nid];
    #pragma omp parallel for
    for (int fid = 0; fid < par_hist.size(); fid++) {
      if (par_hist[fid] != nullptr) {
        par_hist[fid]->SubtractBy(*sib_hist[fid]);
      }
    }
    node_hists[nid] = node_hists[par_nid];
    node_hists[par_nid] = nullptr;
  }

  inline NodeHist& get(nid_t nid) {
    return *node_hists[nid];
  }

  inline void remove(nid_t nid) {
    if (nid < node_hists.size() && node_hists[nid] != nullptr) {
      Free(node_hists[nid]);
      node_hists[nid] = nullptr;
    }
  }

private:

  // build with consecutive instance ids, 
  // for root node without instance sampling
  std::shared_ptr<NodeHist> 
  SparseBuild0(const CompactDataset<float, int>& dataset, 
               const InstanceInfo& ins_info, const FeatureInfo& feat_info, 
               const NodeIndexer& node_indexer, int class_id = -1) {
    uint32_t n_ins = node_indexer.get_node_size(0);
    int n_thr = OMP_GET_NUM_THREADS();
    if (n_thr == 1 || n_ins <= MIN_WORKSET_SIZE) {
      // build with one thread
      std::shared_ptr<NodeHist> node_hist = GetOrAlloc(ins_info, feat_info);
      SparseBuild0Func(dataset, ins_info, feat_info, 
        0, node_indexer.get_node_size(0), *node_hist, class_id);
      return node_hist;
    } else {
      // build with multi-threading
      // calc worksets
      uint32_t workset_size = MAX(MIN_WORKSET_SIZE, 
        DIVUP(n_ins, n_thr * n_thr));
      uint32_t n_worksets = DIVUP(n_ins, workset_size);
      n_thr = MIN(n_thr, n_worksets);
      // alloc workspace for each thread
      std::vector<std::shared_ptr<NodeHist>> workspace(n_thr);
      for (int i = 0; i < n_thr; i++) 
        workspace[i] = GetOrAlloc(ins_info, feat_info);
      // parallel build hist
      #pragma omp parallel for schedule(dynamic, 1) num_threads(n_thr)
      for (int wid = 0; wid < n_worksets; wid++) {
        int tid = OMP_GET_THREAD_ID();
        uint32_t from = wid * workset_size;
        uint32_t until = MIN(from + workset_size, n_ins);
        SparseBuild0Func(dataset, ins_info, feat_info, 
          from, until, *workspace[tid], class_id);
      }
      // merge hists of all threads
      NodeHist& merged = *workspace[0];
      #pragma omp parallel for schedule(dynamic)
      for (int fid = 0; fid < merged.size(); fid++) {
        for (int i = 1; i < n_thr; i++) {
          if (merged[fid] != nullptr) {
            merged[fid]->PlusBy(*(workspace[i]->at(fid)));
          }
        }
      }
      // free workspace
      for (int i = 1; i < n_thr; i++)
        Free(workspace[i]);
      return workspace[0];
    }
  }
  
  // build given an instance id list, 
  // for non-root nodes or root node with instance sampling
  std::shared_ptr<NodeHist> 
  SparseBuild1(nid_t nid, const CompactDataset<float, int>& dataset, 
               const InstanceInfo& ins_info, const FeatureInfo& feat_info, 
               const NodeIndexer& node_indexer, int class_id = -1) {
    uint32_t node_start = node_indexer.get_node_start(nid);
    uint32_t node_end = node_indexer.get_node_end(nid);
    uint32_t node_size = node_end - node_start;
    const auto& node_to_ins = node_indexer.get_node_to_ins();
    int n_thr = OMP_GET_NUM_THREADS();
    if (n_thr == 1 || node_size <= MIN_WORKSET_SIZE) {
      // build with one thread
      std::shared_ptr<NodeHist> node_hist = GetOrAlloc(ins_info, feat_info);
      SparseBuild1Func(dataset, ins_info, feat_info, node_to_ins, 
        node_start, node_end, *node_hist, class_id);
      return node_hist;
    } else {
      // build with multi-threading
      // calc worksets
      uint32_t workset_size = MAX(MIN_WORKSET_SIZE, 
        DIVUP(node_size, n_thr * n_thr));
      uint32_t n_worksets = DIVUP(node_size, workset_size);
      n_thr = MIN(n_thr, n_worksets);
      // alloc workspace for each thread
      std::vector<std::shared_ptr<NodeHist>> workspace(n_thr);
      for (int i = 0; i < n_thr; i++) 
        workspace[i] = GetOrAlloc(ins_info, feat_info);
      // parallel build hist
      #pragma omp parallel for schedule(dynamic, 1) num_threads(n_thr)
      for (int wid = 0; wid < n_worksets; wid++) {
        int tid = OMP_GET_THREAD_ID();
        uint32_t from = node_start + wid * workset_size;
        uint32_t until = MIN(from + workset_size, node_end);
        SparseBuild1Func(dataset, ins_info, feat_info, node_to_ins, 
          from, until, *workspace[tid], class_id);
      }
      // merge hists of all threads
      NodeHist& merged = *workspace[0];
      #pragma omp parallel for schedule(dynamic)
      for (int fid = 0; fid < merged.size(); fid++) {
        for (int i = 1; i < n_thr; i++) {
          if (merged[fid] != nullptr) {
            merged[fid]->PlusBy(*(workspace[i]->at(fid)));
          }
        }
      }
      // free workspace
      for (int i = 1; i < n_thr; i++)
        Free(workspace[i]);
      return workspace[0];
    }
  }

  void SparseBuild0Func(const CompactDataset<float, int>& dataset, 
                        const InstanceInfo& ins_info, 
                        const FeatureInfo& feat_info, 
                        const uint32_t from, const uint32_t until, 
                        NodeHist& node_hist, int class_id = -1) {
    const auto& m1 = ins_info.m1;
    const auto& m2 = ins_info.m2;
    const auto m1_size = ins_info.m1_size;
    const auto m2_size = ins_info.m2_size;
    const auto& is_feat_used = feat_info.is_feat_used;
    if (num_classes == 2) {
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t ins_id = from, val_offset = ins_id * max_dim; 
            ins_id < until; ins_id++, val_offset += max_dim) {
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, m1[ins_id], m2[ins_id]);
            }
          }
        }
      } else {
        for (uint32_t ins_id = from; ins_id < until; ins_id++) {
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, m1[ins_id], m2[ins_id]);
            }
          }
        }
      }
    } else if (multi_tree) {
      // multi-class, using one-vs-rest trees
      ASSERT(class_id >= 0 && class_id < num_classes) 
        << "Error class id: " << class_id;
      uint32_t m_offset = ins_info.num_ins * class_id;
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t ins_id = from, val_offset = ins_id * max_dim; 
            ins_id < until; ins_id++, val_offset += max_dim) {
          auto ins_m1 = m1[m_offset + ins_id];
          auto ins_m2 = m2[m_offset + ins_id];
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, ins_m1, ins_m2);
            }
          }
        }
      } else {
        for (uint32_t ins_id = from; ins_id < until; ins_id++) {
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          auto ins_m1 = m1[m_offset + ins_id];
          auto ins_m2 = m2[m_offset + ins_id];
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, ins_m1, ins_m2);
            }
          }
        }
      }
    } else if (m1_size == m2_size) {
      // multi-class, assuming m2 is diagonal
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t ins_id = from, val_offset = ins_id * max_dim; 
            ins_id < until; ins_id++, val_offset += max_dim) {
          uint32_t m_offset = ins_id * num_classes;
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, m1, m2, m_offset);
            }
          }
        }
      } else {
        for (uint32_t ins_id = from; ins_id < until; ins_id++) {
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          uint32_t m_offset = ins_id * num_classes;
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, m1, m2, m_offset);
            }
          }
        }
      }
    } else {
      // multi-class, m1 and m2 have different size
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t ins_id = from, val_offset = ins_id * max_dim; 
            ins_id < until; ins_id++, val_offset += max_dim) {
          uint32_t m1_offset = ins_id * m1_size;
          uint32_t m2_offset = ins_id * m2_size;
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, m1, m1_offset, m2, m2_offset);
            }
          }
        }
      } else {
        for (uint32_t ins_id = from; ins_id < until; ins_id++) {
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          uint32_t m1_offset = ins_id * m1_size;
          uint32_t m2_offset = ins_id * m2_size;
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, m1, m1_offset, m2, m2_offset);
            }
          }
        }
      }
    }
  }

  void SparseBuild1Func(const CompactDataset<float, int>& dataset, 
                        const InstanceInfo& ins_info, 
                        const FeatureInfo& feat_info, 
                        const std::vector<uint32_t>& ins_ids, 
                        const uint32_t from, const uint32_t until, 
                        NodeHist& node_hist, int class_id = -1) {
    const auto& m1 = ins_info.m1;
    const auto& m2 = ins_info.m2;
    const auto m1_size = ins_info.m1_size;
    const auto m2_size = ins_info.m2_size;
    const auto& is_feat_used = feat_info.is_feat_used;
    if (num_classes == 2) {
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t val_offset = ins_id * max_dim;
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, m1[ins_id], m2[ins_id]);
            }
          }
        }
      } else {
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, m1[ins_id], m2[ins_id]);
            }
          }
        }
      }
    } else if (multi_tree) {
      // multi-class, using one-vs-rest trees
      ASSERT(class_id >= 0 && class_id < num_classes) 
        << "Error class id: " << class_id;
      uint32_t m_offset = ins_info.num_ins * class_id;
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t val_offset = ins_id * max_dim;
          auto ins_m1 = m1[m_offset + ins_id];
          auto ins_m2 = m2[m_offset + ins_id];
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, ins_m1, ins_m2);
            }
          }
        }
      } else {
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          auto ins_m1 = m1[m_offset + ins_id];
          auto ins_m2 = m2[m_offset + ins_id];
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, ins_m1, ins_m2);
            }
          }
        }
      }
    } else if (m1_size == m2_size) {
      // multi-class, assuming m2 is diagonal
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t val_offset = ins_id * max_dim;
          uint32_t m_offset = ins_id * num_classes;
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, m1, m2, m_offset);
            }
          }
        }
      } else {
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          uint32_t m_offset = ins_id * num_classes;
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, m1, m2, m_offset);
            }
          }
        }
      }
    } else {
      // multi-class, m1 and m2 have different size
      if (dataset.is_dense()) {
        uint32_t max_dim = dataset.get_max_dim();
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t val_offset = ins_id * max_dim;
          uint32_t m1_offset = ins_id * m1_size;
          uint32_t m2_offset = ins_id * m2_size;
          for (uint32_t fid = 0; fid < max_dim; fid++) {
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(val_offset + fid);
              node_hist[fid]->Accumulate(bin_id, m1, m1_offset, m2, m2_offset);
            }
          }
        }
      } else {
        for (uint32_t i = from; i < until; i++) {
          uint32_t ins_id = ins_ids[i];
          uint32_t index_start = (ins_id == 0) ? 0 \
            : dataset.get_index_end(ins_id - 1);
          uint32_t index_end = dataset.get_index_end(ins_id);
          uint32_t m1_offset = ins_id * m1_size;
          uint32_t m2_offset = ins_id * m2_size;
          for (uint32_t j = index_start; j < index_end; j++) {
            int fid = dataset.get_indice(j);
            if (is_feat_used[fid]) {
              int bin_id = dataset.get_value(j);
              node_hist[fid]->Accumulate(bin_id, m1, m1_offset, m2, m2_offset);
            }
          }
        }
      }
    }
  }

  void FillDefaultBins(const FeatureInfo& feat_info, 
                       const GradPair& node_gp, NodeHist& node_hist) {
    const auto& default_bins = feat_info.default_bins;
    #pragma omp parallel for
    for (int fid = 0; fid < node_hist.size(); fid++) {
      std::shared_ptr<Histogram>& hist = node_hist[fid];
      if (hist != nullptr) {
        hist->FillRemain(node_gp, default_bins[fid]);
      }
    }
  }

  std::shared_ptr<NodeHist> 
  GetOrAlloc(const InstanceInfo& ins_info, const FeatureInfo& feat_info) {
    if (avail_hist == 0) {
      int num_feat = feat_info.num_feat;
      const auto& is_feat_used = feat_info.is_feat_used;
      const auto& num_bins = feat_info.num_bins;
      uint32_t m1_size = ins_info.m1_size, m2_size = ins_info.m2_size;
      if (num_classes == 2 || multi_tree)
        m1_size = m2_size = 1;
      std::shared_ptr<NodeHist> ret(new NodeHist(num_feat, nullptr));
      for (int fid = 0; fid < num_feat; fid++) {
        if (is_feat_used[fid]) {
          (*ret)[fid] = std::shared_ptr<Histogram>(new Histogram(
            num_bins[fid], m1_size, m2_size));
        }
      }
      return ret;
    } else {
      std::shared_ptr<NodeHist> ret = hist_store[avail_hist - 1];
      hist_store[avail_hist - 1] = nullptr;
      avail_hist--;
      return ret;
    }
  }

  void Free(std::shared_ptr<NodeHist> node_hist) {
    for (int i = 0; i < node_hist->size(); i++) {
      if (node_hist->at(i) != nullptr)
        node_hist->at(i)->clear();
    }
    hist_store[avail_hist] = node_hist;
    avail_hist += 1;
  }

  int num_classes;
  bool multi_tree;
  std::vector<std::shared_ptr<NodeHist>> node_hists;
  std::vector<std::shared_ptr<NodeHist>> hist_store;
  int avail_hist;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_HELPER_HIST_MANAGER_H_
