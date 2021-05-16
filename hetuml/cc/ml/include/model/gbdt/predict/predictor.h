#ifndef __HETU_ML_MODEL_GBDT_PREDICT_PREDICTOR_H_
#define __HETU_ML_MODEL_GBDT_PREDICT_PREDICTOR_H_

#include "common/logging.h"
#include "data/dataset.h"
#include "model/common/mlbase.h"
#include "model/gbdt/model.h"
#include "model/gbdt/helper/node_indexer.h"
#include <memory>

namespace hetu { 
namespace ml {
namespace gbdt {

template <typename Val>
class GBDTPredictor {
public:
  const float learning_rate;
  GBDTPredictor(const GBDTModel& model)
  : model(&model), learning_rate(model.get_param().learning_rate) {
    //
  }

  inline void Predict(std::vector<label_t>& ret, 
                      const DataMatrix<Val>& features, 
                      size_t start_id, size_t end_id) {
    // 
  }

private:
  template<typename T>
  inline const label_t predict_scalar(const DenseVector<Val>& x) const {
    float pred = init_preds[0];
    for (const auto& tree : trees) {
      pred += this->learning_rate * tree->predict_scalar(x);
    }
    return pred;
  }

  /*
  
  template<typename T>
  inline const float predict_scalar(const AVector<T>* x) const {
    if (x->is_dense)
      return predict_scalar((const DenseVector<T>&) *x);
    else
      return predict_scalar((const SparseVector<T>&) *x);
  }

  

  template<typename T>
  inline const float predict_scalar(const SparseVector<T>& x) const {
    float lr = param->learning_rate;
    float pred = init_preds[0];
    for (const auto& tree : trees) {
      pred += lr * tree->predict_scalar(x);
    }
    return pred;
  }

  template<typename T>
  inline void predict_vector(const AVector<T>* x, 
      std::vector<float>& preds) const {
    if (x->is_dense)
      predict_vector((const DenseVector<T>&) *x, preds);
    else
      predict_vector((const SparseVector<T>&) *x, preds);
  }

  template<typename T>
  inline void predict_vector(const DenseVector<T>& x, 
      std::vector<float>& preds) const {
    int num_classes = param->num_label;
    float lr = param->learning_rate;
    std::copy(init_preds.begin(), init_preds.end(), preds.begin());
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
  }

  template<typename T>
  inline void predict_vector(const SparseVector<T>& x, 
      std::vector<float>& preds) const {
    int num_classes = param->num_label;
    float lr = param->learning_rate;
    std::copy(init_preds.begin(), init_preds.end(), preds.begin());
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
  }
  
  
  */

  ~GBDTPredictor() {}

private:
  const GBDTModel* const model;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_PREDICT_PREDICTOR_H_
