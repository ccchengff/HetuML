#ifndef __HETU_ML_MODEL_KNN_KNN_H_
#define __HETU_ML_MODEL_KNN_KNN_H_

#include "model/common/mlbase.h"
#include <queue>

namespace hetu { 
namespace ml {
namespace knn {

namespace KNNConf {
  // number of labels
  static const std::string NUM_LABEL = "NUM_LABEL";
  static const int DEFAULT_NUM_LABEL = 2;
  // number of neighbors
  static const std::string NUM_NEIGHBOR = "NUM_NEIGHBOR";
  static const int DEFAULT_NUM_NEIGHBOR = 5;

  static std::vector<std::string> meaningful_keys() {
    return {
      NUM_LABEL, 
      NUM_NEIGHBOR
    };
  }

  static Args default_args() { 
    return {
      { NUM_LABEL, std::to_string(DEFAULT_NUM_LABEL) }, 
      { NUM_NEIGHBOR, std::to_string(DEFAULT_NUM_NEIGHBOR) }
    };
  }
} // KNNConf

class KNNParam : public MLParam {
public:
  KNNParam(const Args& args = {}, 
           const Args& default_args = KNNConf::default_args())
  : MLParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return KNNConf::meaningful_keys();
  }

  int num_label;
  int num_neighbor;
  std::vector<std::string> metrics;
private:
  inline void InitAndCheckParam() override {
    this->num_label = argparse::Get<int>(
      this->all_args, KNNConf::NUM_LABEL);
    ASSERT_GT(this->num_label, 0) 
      << "Invalid number of labels: " << this->num_label;
    this->num_neighbor = argparse::Get<int>(
      this->all_args, KNNConf::NUM_NEIGHBOR);
    ASSERT_GT(this->num_neighbor, 0) 
      << "Invalid number of neighbors: " << this->num_neighbor;
  }
};

template <typename Val>
class KNN : public MLBase<KNNParam> {
public:
  inline KNN(const Args& args = {}): MLBase<KNNParam>(args) {}

  inline ~KNN() {}

  inline void Predict(std::vector<label_t>& ret, 
                      const Dataset<label_t, Val>& labeled_data, 
                      const DataMatrix<Val>& unlabeled_data) {
    size_t num_labeled = labeled_data.get_num_instances();
    size_t num_unlabeled = unlabeled_data.get_num_instances();  
    size_t num_label = this->params->num_label;
    size_t num_neighbor = this->params->num_neighbor;
    
    ASSERT(!labeled_data.is_dense()) 
      << "Currently we only support sparse features for KNN models";
    ASSERT(!unlabeled_data.is_dense())
      << "Currently we only support sparse features for KNN models";
    ASSERT_GT(num_labeled, 0) 
      << "No labeled instances are provided";
    ASSERT_GT(num_unlabeled, 0) 
      << "No unlabeled instances are provided";
    ASSERT_GT(num_labeled, num_neighbor) 
      << "Only " << num_labeled << " labeled instances are provided, " 
      << "which is not greater than number of neighbors " << num_neighbor;

    TIK(predict);
    HML_LOG_INFO << "Start prediction of " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;
    ret.resize(num_unlabeled * num_label);
    for (size_t i = 0; i < num_unlabeled; i++) {
      const auto& unlabeled_ins = unlabeled_data.get_sparse_feature(i);
      // compute distance and get nearest neighbors
      std::priority_queue<std::tuple<Val, size_t>> neighbors;
      for (size_t j = 0; j < num_labeled; j++) {
        const auto& labeled_ins = labeled_data.get_sparse_feature(j);
        Val dist = KNN<Val>::CalculateDistance(unlabeled_ins, labeled_ins);
        neighbors.push(std::make_tuple(dist, j));
        if (neighbors.size() > num_neighbor)
          neighbors.pop();
      }
      // voting
      size_t offset = i * num_label;
      while (!neighbors.empty()) {
        auto& temp = neighbors.top();
        neighbors.pop();
        auto label = static_cast<size_t>(labeled_data.get_label(
          std::get<1>(temp)));
        ret[offset + label] += 1.0 / num_label;
      }
    }
    TOK(predict);
    HML_LOG_INFO << "Prediction of " << this->name() << " model"
      << " cost " << COST_MSEC(predict) << " ms";
  }

  inline std::vector<Val>
  Evaluate(const Dataset<label_t, Val>& labeled_data, 
           const Dataset<label_t, Val>& eval_data, 
           const std::vector<std::string>& metrics) {
    std::vector<Val> res;
    res.reserve(metrics.size());
    auto num_eval = eval_data.get_num_instances();
    if (num_eval > 0) {
      std::vector<label_t> preds;
      const auto& labels = eval_data.get_labels();
      this->Predict(preds, labeled_data, eval_data);
      for (const auto& metric_name : metrics) {
        const auto* eval_metric = \
          MetricFactory::GetEvalMetric<Val, label_t, Val>(
            metric_name, false);
        Val metric = eval_metric->EvalMulti(preds.data(), labels.data(), 
          this->params->num_label, num_eval);
        res.push_back(metric);
      }
    } else {
      res.resize(metrics.size(), (Val) 0);
    }
    return std::move(res);
  }

  inline void LoadFromStream(std::istream& is) override {
    // do nothing
  }

  inline void DumpToStream(std::ostream& os) override {
    // do nothing
  }

  inline const char* name() const override { return "KNearestNeighbors"; }

  inline int get_num_label() const { return this->params->num_label; }

private:
  inline static Val CalculateDistance(const SparseVector<Val>& x, 
                                      const SparseVector<Val>& y) {
    Val dist = 0;
    size_t i = 0, j = 0;  
    while (i < x.nnz && j < y.nnz) {
      if (x.indices[i] < y.indices[j]) {
        dist += SQUARE(x.values[i]);
        i++;
      } else if (x.indices[i] > y.indices[j]) {
        dist += SQUARE(y.values[j]);
        j++;
      } else {
        dist += SQUARE(x.values[i] - y.values[j]);
        i++;
        j++;
      }
    }
    while (i < x.nnz) {
      dist += SQUARE(x.values[i]);
      i++;
    }
    while (j < y.nnz) {
      dist += SQUARE(y.values[j]);
      j++;
    }
    return dist;
  }
};

} // namespace knn
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_KNN_KNN_H_
