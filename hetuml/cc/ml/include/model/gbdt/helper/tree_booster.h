#ifndef __HETU_ML_MODEL_GBDT_HELPER_TREE_BOOSTER_H_
#define __HETU_ML_MODEL_GBDT_HELPER_TREE_BOOSTER_H_

#include "common/logging.h"
#include "common/threading.h"
#include "model/gbdt/model.h"
#include "model/gbdt/hist/grad_pair.h"
#include "model/gbdt/helper/metadata.h"
#include "objective/loss.h"
#include <memory>

namespace hetu { 
namespace ml {
namespace gbdt {

/******************************************************
 * Boosted objective
 ******************************************************/
class TreeBooster {
public:
  TreeBooster(std::shared_ptr<const GBDTParam> param): param(param) {
    row_classes.reserve(param->num_label);
    col_classes.reserve(param->num_label);
    iso_classes.reserve(param->num_label);
  }

  inline GradPair* CalcGradPairs(const RichLoss<float, float, double>& loss, 
                                 const std::vector<float>& labels, 
                                 InstanceInfo& ins_info) {
    uint32_t num_ins = labels.size();
    const auto& preds = ins_info.predictions;
    auto& ret_grad = ins_info.m1;
    auto& ret_hess = ins_info.m2;
    ASSERT_EQ(preds.size() % num_ins, 0) << "Not divisible";
    ASSERT_EQ(ret_grad.size() % num_ins, 0) << "Not divisible";
    ASSERT_EQ(ret_hess.size() % num_ins, 0) << "Not divisible";
    int num_classes = param->num_label;
    uint32_t m1_size = ret_grad.size() / num_ins;
    uint32_t m2_size = ret_hess.size() / num_ins;
    if (num_classes == 2) {
      double sum_grad = 0, sum_hess = 0;
      #pragma omp parallel for reduction(+:sum_grad,sum_hess)
      for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
        double grad = loss.FirstOrderGrad(preds[ins_id], labels[ins_id]);
        double hess = loss.SecondOrderGrad(preds[ins_id], labels[ins_id], grad);
        ret_grad[ins_id] = grad;
        ret_hess[ins_id] = hess;
        sum_grad += grad;
        sum_hess += hess;
      } 
      return new BinaryGradPair(sum_grad, sum_hess);
    } else if (m1_size == m2_size) {
      // multi-class, assuming hessian is diagonal or using one-vs-rest trees
      std::vector<double> sum_grad(num_classes, 0), sum_hess(num_classes, 0);
      const float *pred_ptr = preds.data();
      double *grad_ptr = ret_grad.data(), *hess_ptr = ret_hess.data();
      #pragma omp parallel for reduction(vec_double_plus:sum_grad,sum_hess)
      for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
        const float *pred = pred_ptr + ins_id * num_classes;
        double *grad = grad_ptr + ins_id * num_classes;
        double *hess = hess_ptr + ins_id * num_classes;
        loss.FirstOrderGrad(pred, labels[ins_id], num_classes, grad);
        loss.SecondOrderGradDiag(pred, labels[ins_id], grad, num_classes, hess);
        for (int k = 0; k < num_classes; k++) {
          sum_grad[k] += grad[k];
          sum_hess[k] += hess[k];
        }
      }
      if (param->multi_tree) {
        // co-locate grad & hess for the same class to be cache-friendly
        std::vector<double> saved_grad(ret_grad), saved_hess(ret_hess);
        for (int k = 0; k < num_classes; k++) {
          uint32_t offset = k * num_ins;
          for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
            uint32_t tmp = ins_id * num_classes;
            ret_grad[offset + ins_id] = saved_grad[tmp + k];
            ret_hess[offset + ins_id] = saved_hess[tmp + k];
          }
        }
      }
      return new MultiGradPair(sum_grad, sum_hess);
    } else if (m1_size * (m1_size + 1) == m2_size * 2) {
      // multi-class, using full hessian matrix
      std::vector<double> sum_grad(m1_size, 0), sum_hess(m2_size, 0);
      const float *pred_ptr = preds.data();
      double *grad_ptr = ret_grad.data(), *hess_ptr = ret_hess.data();
      #pragma omp parallel for reduction(vec_double_plus:sum_grad,sum_hess)
      for (uint32_t ins_id = 0; ins_id < num_ins; ins_id++) {
        const float *pred = pred_ptr + ins_id * num_classes;
        double *grad = grad_ptr + ins_id * m1_size;
        double *hess = hess_ptr + ins_id * m2_size;
        loss.FirstOrderGrad(pred, labels[ins_id], num_classes, grad);
        loss.SecondOrderGradFull(pred, labels[ins_id], grad, num_classes, hess);
        for (int k = 0; k < m1_size; k++) 
          sum_grad[k] += grad[k];
        for (int k = 0; k < m2_size; k++) 
          sum_hess[k] += hess[k];
      }
      this->full_hessian = true;
      return new MultiGradPair(sum_grad, sum_hess);
    } else {
      ASSERT(false) << "Sparsified hessian is not supported yet";
      return nullptr;
    }
  }

  inline bool SatisfyWeight(const GradPair& gp) const {
    if (param->min_child_weight == 0) 
      return true;
    else if (!param->IsLeafVector()) 
      return SatisfyWeight((const BinaryGradPair&) gp);
    else 
      return SatisfyWeight((const MultiGradPair&) gp);
  }

  inline bool SatisfyWeight(const BinaryGradPair& binary) const {
    return SatisfyWeight(binary.get_m1(), binary.get_m2());
  }

  inline bool SatisfyWeight(const MultiGradPair& multi) const {
    return SatisfyWeight(multi.get_m1(), multi.get_m2());
  }

  inline double CalcGain(const GradPair& gp) const {
    if (!param->IsLeafVector()) 
      return CalcGain((const BinaryGradPair&) gp);
    else 
      return CalcGain((const MultiGradPair&) gp);
  }

  inline double CalcGain(const BinaryGradPair& binary) const {
    return CalcGain(binary.get_m1(), binary.get_m2());
  }

  inline double CalcGain(const MultiGradPair& multi) const {
    return CalcGain(multi.get_m1(), multi.get_m2());
  }

  inline double CalcWeight(const BinaryGradPair& binary) const {
    return CalcWeight(binary.get_m1(), binary.get_m2());
  }

  inline void CalcWeights(const MultiGradPair& multi, 
                          std::vector<double>& ret) const {
    return CalcWeights(multi.get_m1(), multi.get_m2(), ret);
  }

  inline void CalcNodeGain(GBTNode& node) const {
    node.set_gain(CalcGain(node.get_sum_gp()));
  }

  inline void CalcLeafWeight(GBTNode& leaf) const {
    auto binary = (const BinaryGradPair&) leaf.get_sum_gp();
    double weight = CalcWeight(binary);
    leaf.set_weight(weight);
  }

  inline void CalcLeafWeights(GBTNode& leaf) const {
    auto multi = (const MultiGradPair&) leaf.get_sum_gp();
    std::vector<double> weights(param->num_label);
    CalcWeights(multi, weights);
    leaf.set_weights(weights);
  }

  inline bool SatisfyWeight(double grad, double hess) const {
    return hess > param->min_child_weight;
  }

  inline bool SatisfyWeight(const std::vector<double>& grad, 
                            const std::vector<double>& hess) const {
    // Since hessian matrix is positive, 
    // we have det(hess) <= a11*a22*...*akk, 
    // thus we approximate det(hess) with a11*a22*...*akk
    int num_classes = grad.size();
    if (!this->full_hessian) {
      return Prod<double>(hess, 0, num_classes) > param->min_child_weight;
    } else {
      return ProdDiag<double>(hess, num_classes) > param->min_child_weight;
    }
  }

  inline double CalcGain(double grad, double hess) const {
    double max_leaf_weight = param->max_leaf_weight;
    double reg_alpha = param->reg_alpha;
    double reg_lambda = param->reg_lambda;

    if (max_leaf_weight == 0) {
      if (reg_alpha == 0) {
        return (grad / (hess + reg_lambda)) * grad;
      } else {
        double thr_grad = ThresholdL1(grad, reg_alpha);
        return (thr_grad / (hess + reg_lambda)) * thr_grad;
      }
    } else {
      double w = CalcWeight(grad, hess);
      double ret = grad * w + 0.5 * (hess + reg_lambda) * w * w;
      if (reg_alpha == 0)
        return -2 * ret;
      else
        return -2 * (ret + reg_alpha * ABS(w));
    }
  }

  double CalcGain(const std::vector<double>& grad, 
                  const std::vector<double>& hess) const {
    int num_classes = grad.size();
    double max_leaf_weight = param->max_leaf_weight;
    double reg_alpha = param->reg_alpha;
    double reg_lambda = param->reg_lambda;
    
    double gain = 0;
    if (hess.size() == num_classes) {
      if (reg_alpha == 0) {
        for (int k = 0; k < num_classes; k++)
          gain += (grad[k] / (hess[k] + reg_lambda)) * grad[k];
      } else {
        for (int k = 0; k < num_classes; k++) {
          double thr_grad = ThresholdL1(grad[k], reg_alpha);
          gain += (thr_grad / (hess[k] + reg_lambda)) * thr_grad;
        }
      }
    } else if (this->full_hessian) {
      std::vector<double> hess_plus_lambda(hess);
      AddDiagonal<double>(hess_plus_lambda, num_classes, reg_lambda);
      std::vector<double> tmp(num_classes);
      SolveLinearSystemWithCholeskyDecomposition<double>(
        hess_plus_lambda.data(), grad.data(), num_classes, tmp.data());
      gain = VecDot<double>(grad, tmp);
      AddDiagonal<double>(hess_plus_lambda, num_classes, -reg_lambda);
    } else {
      for (int k = 0; k < row_classes.size(); k++) {
        int i = row_classes[k], j = col_classes[k];
        double G_i = grad[i];
        double G_j = grad[j];
        double H_ii = hess[i] + reg_lambda;
        double H_jj = hess[j] + reg_lambda;
        double H_ij = hess[num_classes + k];
        double tmp = H_ii * H_jj - H_ij * H_ij;
        double Inv_ii = H_jj / tmp;
        double Inv_jj = H_ii / tmp;
        double Inv_ij = -H_ij / tmp;
        gain += G_i * Inv_ii * G_i + G_j * Inv_jj * G_j \
          + 2 * G_i * Inv_ij * G_j;
      }
      for (int k = 0; k < iso_classes.size(); k++) {
        int i = iso_classes[k];
        double G_i = grad[i];
        double H_ii = hess[i] + reg_lambda;
        gain += G_i / H_ii * G_i;
      }
    }
    return gain / num_classes;
  }

  inline double CalcWeight(double grad, double hess) const {
    if (!SatisfyWeight(grad, hess)) return 0.0;
  
    double max_leaf_weight = param->max_leaf_weight;
    double reg_alpha = param->reg_alpha;
    double reg_lambda = param->reg_lambda;

    double w;
    if (reg_alpha == 0) 
      w = -grad / (hess + reg_lambda);
    else 
      w = -ThresholdL1(grad, reg_alpha) / (hess + reg_lambda);
    if (max_leaf_weight != 0) {
      if (w > max_leaf_weight)
        w = max_leaf_weight;
      else if (w < -max_leaf_weight)
        w = -max_leaf_weight;
    }
    return w;
  }

  void CalcWeights(const std::vector<double>& grad, 
                   const std::vector<double>& hess, 
                   std::vector<double>& ret) const {
    int num_classes = grad.size();
    double reg_alpha = param->reg_alpha;
    double reg_lambda = param->reg_lambda;

    if (hess.size() == num_classes) {
      if (reg_alpha == 0) {
        for (int k = 0; k < num_classes; k++) 
          ret[k] = -grad[k] / (hess[k] + reg_lambda);
      } else {
        for (int k = 0; k < num_classes; k++) 
          ret[k] = -ThresholdL1(grad[k], reg_alpha) / (hess[k] + reg_lambda);
      }
    } else if (this->full_hessian) {
      std::vector<double> hess_plus_lambda(hess);
      AddDiagonal<double>(hess_plus_lambda, num_classes, reg_lambda);
      SolveLinearSystemWithCholeskyDecomposition<double>(
        hess_plus_lambda.data(), grad.data(), num_classes, ret.data());
      for (int k = 0; k < num_classes; k++) ret[k] *= -1;
      AddDiagonal<double>(hess_plus_lambda, num_classes, -reg_lambda);
    } else {
      for (int k = 0; k < row_classes.size(); k++) {
        int i = row_classes[k], j = col_classes[k];
        double G_i = grad[i];
        double G_j = grad[j];
        if (reg_alpha != 0) {
          G_i = ThresholdL1(G_i, reg_alpha);
          G_j = ThresholdL1(G_j, reg_alpha);
        }
        double H_ii = hess[i] + reg_lambda;
        double H_jj = hess[j] + reg_lambda;
        double H_ij = hess[num_classes + k];
        double tmp = H_ii * H_jj - H_ij * H_ij;
        double Inv_ii = H_jj / tmp;
        double Inv_ij = -H_ij / tmp;
        double Inv_jj = H_ii / tmp;
        ret[i] = -(G_i * Inv_ii + G_j * Inv_ij);
        ret[j] = -(G_j * Inv_jj + G_i * Inv_ij);
      }
      for (int k = 0; k < iso_classes.size(); k++) {
        int i = iso_classes[k];
        double G_i = grad[i];
        if (reg_alpha != 0) 
          G_i = ThresholdL1(G_i, reg_alpha);
        double H_ii = hess[i] + reg_lambda;
        ret[i] = -G_i / H_ii;
      }
    }
  }

private:
  std::shared_ptr<const GBDTParam> param;
  bool full_hessian = false;
  std::vector<int> row_classes, col_classes, iso_classes;
};


} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_HELPER_TREE_BOOSTER_H_
