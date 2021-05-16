#ifndef __HETU_ML_MODEL_GBDT_HELPER_SPLIT_FINDER_H_
#define __HETU_ML_MODEL_GBDT_HELPER_SPLIT_FINDER_H_

#include "common/logging.h"
#include "common/threading.h"
#include "model/gbdt/model.h"
#include "model/gbdt/hist/histogram.h"
#include "model/gbdt/helper/metadata.h"
#include "model/gbdt/helper/tree_booster.h"
#include <memory>

namespace hetu { 
namespace ml {
namespace gbdt {

/******************************************************
 * Finder for the best split
 ******************************************************/
class SplitFinder {
public:

  SplitFinder(int num_classes, bool multi_tree, 
              float min_split_gain, float reg_lambda)
  : num_classes(num_classes), multi_tree(multi_tree), 
  min_split_gain(min_split_gain), reg_lambda(reg_lambda) {}

  std::shared_ptr<GBTSplit> 
  FindBestSplit(const FeatureInfo& feat_info, const NodeHist& node_hist, 
                const GradPair& node_gp, float node_gain, 
                const TreeBooster& booster) const {
    const auto& splits = feat_info.splits;
    const auto& num_bins = feat_info.num_bins;
    std::vector<std::shared_ptr<GBTSplit>> buffer(node_hist.size());
    #pragma omp parallel for schedule(dynamic)
    for (int fid = 0; fid < node_hist.size(); fid++) {
      if (node_hist[fid] != nullptr) {
        buffer[fid] = FindBestSplitPointOfOneFeature(*node_hist[fid], fid, 
          num_bins[fid], splits[fid], node_gp, node_gain, booster);
      } else {
        buffer[fid] = nullptr;
      }
    }
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
    return best_fid >= 0 ? buffer[best_fid] : nullptr;
  }

  std::shared_ptr<GBTSplit> 
  FindBestSplitPointOfOneFeature(const Histogram& hist, int fid, 
                                 const int num_bins, 
                                 const std::vector<float>& splits, 
                                 const GradPair& node_gp, float node_gain, 
                                 const TreeBooster& booster) const {
    auto& m1 = hist.get_m1();
    auto& m2 = hist.get_m2();
    uint32_t m1_size = hist.get_m1_size();
    uint32_t m2_size = hist.get_m2_size();
    return FindBestSplitPointOfOneFeature(m1, m1_size, m2, m2_size, 
      fid, num_bins, splits, node_gp, node_gain, booster);
  }

  std::shared_ptr<GBTSplit> 
  FindBestSplitPointOfOneFeature(const std::vector<double>& m1, 
                                 uint32_t m1_size, 
                                 const std::vector<double>& m2, 
                                 uint32_t m2_size, 
                                 int fid, const int num_bins, 
                                 const std::vector<float>& splits, 
                                 const GradPair& node_gp, float node_gain, 
                                 const TreeBooster& booster) const {
    if (num_classes == 2 || multi_tree) {
      // binary-class or multi-class with one-ve-rest trees
      double best_left_m1, best_left_m2;
      double best_right_m1, best_right_m2;
      int best_bin_id = -1;
      float best_gain = min_split_gain - LARGER_EPSILON;
      double left_m1 = 0;
      double left_m2 = 0;
      double right_m1 = ((const BinaryGradPair&) node_gp).get_m1();
      double right_m2 = ((const BinaryGradPair&) node_gp).get_m2();
      for (int bin_id = 0; bin_id < num_bins - 1; bin_id++) {
        left_m1 += m1[bin_id];
        left_m2 += m2[bin_id];
        right_m1 -= m1[bin_id];
        right_m2 -= m2[bin_id];
        if (booster.SatisfyWeight(left_m1, left_m2) 
            && booster.SatisfyWeight(right_m1, right_m2)) {
          float left_gain = booster.CalcGain(left_m1, left_m2);
          float right_gain = booster.CalcGain(right_m1, right_m2);
          float split_gain = left_gain + right_gain - node_gain - reg_lambda;
          if (split_gain > best_gain) {
            best_left_m1 = left_m1;
            best_left_m2 = left_m2;
            best_right_m1 = right_m1;
            best_right_m2 = right_m2;
            best_bin_id = bin_id;
            best_gain = split_gain;
          }
        }
      }
      if (best_bin_id >= 0 && best_gain > min_split_gain) {
        std::unique_ptr<GradPair> best_left(
          new BinaryGradPair(best_left_m1, best_left_m2));
        std::unique_ptr<GradPair> best_right(
          new BinaryGradPair(best_right_m1, best_right_m2));
        std::unique_ptr<SplitEntry> sp(
          new SplitPoint(fid, splits[best_bin_id + 1], best_gain));
        return std::shared_ptr<GBTSplit>(
          new GBTSplit(sp, best_left, best_right));
      }
    } else if (m1_size == m2_size) {
      // multi-class assuming m2 is diagonal
      std::vector<double> best_left_m1(num_classes), best_left_m2(num_classes);
      std::vector<double> best_right_m1(num_classes), best_right_m2(num_classes);
      int best_bin_id = -1;
      float best_gain = min_split_gain - LARGER_EPSILON;
      std::vector<double> left_m1(num_classes);
      std::vector<double> left_m2(num_classes);
      std::vector<double> right_m1(((const MultiGradPair&) node_gp).get_m1());
      std::vector<double> right_m2(((const MultiGradPair&) node_gp).get_m2());
      for (int bin_id = 0, offset = 0; bin_id < num_bins - 1; 
          bin_id++, offset += num_classes) {
        for (int k = 0; k < num_classes; k++) {
          left_m1[k] += m1[offset + k];
          left_m2[k] += m2[offset + k];
          right_m1[k] -= m1[offset + k];
          right_m2[k] -= m2[offset + k];
        }
        if (booster.SatisfyWeight(left_m1, left_m2) 
            && booster.SatisfyWeight(right_m1, right_m2)) {
          float left_gain = booster.CalcGain(left_m1, left_m2);
          float right_gain = booster.CalcGain(right_m1, right_m2);
          float split_gain = left_gain + right_gain - node_gain - reg_lambda;
          if (split_gain > best_gain) {
            std::copy(left_m1.begin(), left_m1.end(), best_left_m1.begin());
            std::copy(left_m2.begin(), left_m2.end(), best_left_m2.begin());
            std::copy(right_m1.begin(), right_m1.end(), best_right_m1.begin());
            std::copy(right_m2.begin(), right_m2.end(), best_right_m2.begin());
            best_bin_id = bin_id;
            best_gain = split_gain;
          }
        }
      }
      if (best_bin_id >= 0 && best_gain > min_split_gain) {
        std::unique_ptr<GradPair> best_left(
          new MultiGradPair(best_left_m1, best_left_m2));
        std::unique_ptr<GradPair> best_right(
          new MultiGradPair(best_right_m1, best_right_m2));
        std::unique_ptr<SplitEntry> sp(
          new SplitPoint(fid, splits[best_bin_id + 1], best_gain));
        return std::shared_ptr<GBTSplit>(
          new GBTSplit(sp, best_left, best_right));
      }
    } else {
      // multi-class when m1 and m2 have different sizes
      std::vector<double> best_left_m1(m1_size), best_left_m2(m2_size);
      std::vector<double> best_right_m1(m1_size), best_right_m2(m2_size);
      int best_bin_id = -1;
      float best_gain = min_split_gain - LARGER_EPSILON;
      std::vector<double> left_m1(m1_size);
      std::vector<double> left_m2(m2_size);
      std::vector<double> right_m1(((const MultiGradPair&) node_gp).get_m1());
      std::vector<double> right_m2(((const MultiGradPair&) node_gp).get_m2());
      for (int bin_id = 0, m1_offset = 0, m2_offset = 0; bin_id < num_bins - 1; 
          bin_id++, m1_offset += m1_size, m2_offset += m2_size) {
        for (int k = 0; k < m1_size; k++) {
          left_m1[k] += m1[m1_offset + k];
          right_m1[k] -= m1[m1_offset + k];
        }
        for (int k = 0; k < m2_size; k++) {
          left_m2[k] += m2[m2_offset + k];
          right_m2[k] -= m2[m2_offset + k];
        }
        if (booster.SatisfyWeight(left_m1, left_m2) 
            && booster.SatisfyWeight(right_m1, right_m2)) {
          float left_gain = booster.CalcGain(left_m1, left_m2);
          float right_gain = booster.CalcGain(right_m1, right_m2);
          float split_gain = left_gain + right_gain - node_gain - reg_lambda;
          if (split_gain > best_gain) {
            std::copy(left_m1.begin(), left_m1.end(), best_left_m1.begin());
            std::copy(left_m2.begin(), left_m2.end(), best_left_m2.begin());
            std::copy(right_m1.begin(), right_m1.end(), best_right_m1.begin());
            std::copy(right_m2.begin(), right_m2.end(), best_right_m2.begin());
            best_bin_id = bin_id;
            best_gain = split_gain;
          }
        }
      }
      if (best_bin_id >= 0 && best_gain > min_split_gain) {
        std::unique_ptr<GradPair> best_left(
          new MultiGradPair(best_left_m1, best_left_m2));
        std::unique_ptr<GradPair> best_right(
          new MultiGradPair(best_right_m1, best_right_m2));
        std::unique_ptr<SplitEntry> sp(
          new SplitPoint(fid, splits[best_bin_id + 1], best_gain));
        return std::shared_ptr<GBTSplit>(
          new GBTSplit(sp, best_left, best_right));
      }
    }
    return nullptr;
  }

private:
  int num_classes;
  bool multi_tree;
  float min_split_gain;
  float reg_lambda;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_HELPER_SPLIT_FINDER_H_
