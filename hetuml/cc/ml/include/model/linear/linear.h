#ifndef __HETU_ML_MODEL_LINEAR_LINEAR_H_
#define __HETU_ML_MODEL_LINEAR_LINEAR_H_

#include "model/common/mlbase.h"

namespace hetu { 
namespace ml {
namespace linear {

namespace LinearConf {
  // number of epochs
  static const std::string NUM_EPOCH = "NUM_EPOCH";
  static const int DEFAULT_NUM_EPOCH = 10;
  // batch size
  static const std::string BATCH_SIZE = "BATCH_SIZE";
  static const int DEFAULT_BATCH_SIZE = 1000;
  // learning rate
  static const std::string LEARNING_RATE = "LEARNING_RATE";
  static const float DEFAULT_LEARNING_RATE = 0.1f;
  // L1 regularization
  static const std::string L1_REG = "L1_REG";
  static const float DEFAULT_L1_REG = 0.0f;
  // L2 regularization
  static const std::string L2_REG = "L2_REG";
  static const float DEFAULT_L2_REG = 0.0f;
  // loss function
  static const std::string LOSS = "LOSS";
  // evaluation metric
  static const std::string METRICS = "METRICS";

  static Args AddObjectiveIfNotExists(const Args& args, 
                                      const std::string& loss_type, 
                                      const std::string& metric_type) {
    Args res = args;
    if (res.find(LOSS) == res.end()) {
      res[LOSS] = loss_type;
    }
    if (res.find(METRICS) == res.end()) {
      res[METRICS] = metric_type;
    }
    return res;
  }

  static std::vector<std::string> meaningful_keys() {
    return {
      NUM_EPOCH, 
      BATCH_SIZE, 
      LEARNING_RATE, 
      L1_REG, 
      L2_REG, 
      LOSS, 
      METRICS
    };
  }

  static Args default_args() { 
    return {
      { NUM_EPOCH, std::to_string(DEFAULT_NUM_EPOCH) }, 
      { BATCH_SIZE, std::to_string(DEFAULT_BATCH_SIZE) }, 
      { LEARNING_RATE, std::to_string(DEFAULT_LEARNING_RATE) }, 
      { L1_REG, std::to_string(DEFAULT_L1_REG) }, 
      { L2_REG, std::to_string(DEFAULT_L2_REG) }, 
      { LOSS, "not-provided" }, 
      { METRICS, "" }
    };
  }
} // LinearConf

class LinearParam : public MLParam {
public:
  LinearParam(const Args& args = {}, 
              const Args& default_args = LinearConf::default_args())
  : MLParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return LinearConf::meaningful_keys();
  }

  int num_epoch;
  int batch_size;
  float learning_rate;
  float l1_reg;
  float l2_reg;
  std::string loss;
  std::vector<std::string> metrics;
protected:
  inline void InitAndCheckParam() override {
    this->num_epoch = argparse::Get<int>(
      this->all_args, LinearConf::NUM_EPOCH);
    this->batch_size = argparse::Get<int>(
      this->all_args, LinearConf::BATCH_SIZE);
    this->learning_rate = argparse::Get<float>(
      this->all_args, LinearConf::LEARNING_RATE);
    this->l1_reg = argparse::Get<float>(
      this->all_args, LinearConf::L1_REG);
    this->l2_reg = argparse::Get<float>(
      this->all_args, LinearConf::L2_REG);
    this->loss = argparse::Get<std::string>(
      this->all_args, LinearConf::LOSS);
    this->metrics = argparse::GetVector<std::string>(
      this->all_args, LinearConf::METRICS);
    // check whether loss and metrics are properly set
    ASSERT(LossFactory::GetLoss<label_t>(this->loss) != nullptr)
      << "Undefined loss: " << this->loss;
    for (const auto& metric : this->metrics) {
      const auto* eval_metric = \
        MetricFactory::GetEvalMetric<label_t, label_t, label_t>(metric, true);
      ASSERT(eval_metric != nullptr) << "Undefined metric: " << metric;
    }
  }
};

template <typename Val>
class Linear : public SupervisedMLBase<Val, LinearParam> {
public:
  inline Linear(const Args& args = {})
  : SupervisedMLBase<Val, LinearParam>(args) {}

  inline ~Linear() {}

  inline void Fit(const Dataset<label_t, Val>& train_data, 
                  const Dataset<label_t, Val>& valid_data = {}) override {
    ASSERT(!train_data.is_dense()) 
      << "Currently we only support sparse features for Linear models";
    ASSERT(!valid_data.is_dense() || valid_data.get_num_instances() == 0)
      << "Currently we only support sparse features for Linear models";
    if (this->use_neg_y()) { // LogReg or SVM
      train_data.CheckBinaryLabels(this->use_neg_y());
      valid_data.CheckBinaryLabels(this->use_neg_y());
    }
    
    this->InitModel(train_data.get_max_dim());

    TIK(fit);
    HML_LOG_INFO << "Start to fit " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;
    auto num_train = train_data.get_num_instances();
    auto num_valid = valid_data.get_num_instances();
    const auto& metrics = this->params->metrics;
    for (size_t eid = 0; eid < this->params->num_epoch; eid++) {
      // train one epoch on training set
      auto train_loss = this->FitOneEpoch(train_data);
      HML_LOG_INFO << "Epoch[" << eid << "] Train loss: " 
        << this->params->loss << "[" << train_loss << "]";
      // evaluation on validation set
      if (num_valid > 0 && !metrics.empty()) {
        auto m = this->Evaluate(valid_data, metrics);
        HML_LOG_INFO << "Epoch[" << eid << "] Valid " 
          << MetricFactory::to_string(metrics, m);
      }
    }
    TOK(fit);
    HML_LOG_INFO << "Fit " << this->name() << " model"
      << " cost " << COST_MSEC(fit) << " ms";
  }

  inline void Predict(std::vector<label_t>& ret, 
                      const DataMatrix<Val>& features, 
                      size_t start_id, size_t end_id) override {
    ASSERT(!this->is_empty()) << "Model is empty";
    ASSERT(!features.is_dense()) 
      << "Currently we only support sparse features for Linear models";
    ret.resize(end_id - start_id);
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      const auto& feature = features.get_sparse_feature(ins_id);
      ret[ins_id - start_id] = this->model->dot(feature);
    }
  }
  
  inline void LoadFromStream(std::istream& is) override {
    size_t max_dim;
    is >> max_dim;
    this->InitModel(max_dim);
    for (size_t dim = 0; dim < max_dim; dim++) {
      is >> this->model->values[dim];
    }
  }

  inline void DumpToStream(std::ostream& os) override {
    size_t max_dim = this->get_max_dim();
    os << max_dim << std::endl;
    for (size_t dim = 0; dim < max_dim; dim++)
      os << this->model->values[dim] << std::endl;
  }

  inline bool is_empty() const { 
    return this->model == nullptr; 
  }

  inline size_t get_max_dim() const {
    ASSERT(!this->is_empty()) << "Model is empty";
    return this->model->dim;
  }

  virtual bool is_regression() const { return false; }

protected:
  virtual void InitModel(size_t max_dim) {
    ASSERT_GT(max_dim, 0) << "Illegal number of dimensions: " << max_dim;
    this->model.reset(new DenseVector<Val>(max_dim));
    if (this->is_regression()) {
      // Zero init
      std::fill(this->model->values, this->model->values + max_dim, 0);
    } else {
      // XavierNormal init
      NormalDistribution<Val>(this->model->values, max_dim, 
        0, std::sqrt(2.0 / (max_dim + 1.0)));
    }
  }

  inline virtual Val FitOneEpoch(const Dataset<label_t, Val>& train_data) {
    auto num_train = train_data.get_num_instances();
    auto batch_size = this->params->batch_size;
    auto num_batch = DIVUP(num_train, batch_size);
    Val loss = 0;
    for (size_t bid = 0; bid < num_batch; bid++) {
      size_t start_id = bid * batch_size;
      size_t end_id = MIN(start_id + batch_size, num_train);
      Val batch_loss = this->FitOneBatch(train_data, start_id, end_id);
      loss += batch_loss;
    }
    return loss / num_batch;
  }

  inline virtual Val FitOneBatch(const Dataset<label_t, Val>& train_data, 
                                 size_t start_id, size_t end_id) {
    std::vector<label_t> preds(end_id - start_id);
    this->Predict(preds, train_data, start_id, end_id);
    const auto* loss_func = LossFactory::GetLoss<Val>(this->params->loss);
    auto step = -this->params->learning_rate / (end_id - start_id);
    Val batch_loss = 0;
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      const auto& label = train_data.get_label(ins_id);
      const auto& feature = train_data.get_sparse_feature(ins_id);
      auto pred = preds[ins_id - start_id];
      batch_loss += loss_func->loss(pred, label);
      Val grad_multipler = loss_func->grad(pred, label);
      this->model->axp0(feature, step * grad_multipler);
    }
    return batch_loss / (end_id - start_id);
  }

  std::unique_ptr<DenseVector<Val>> model;
};

template <typename Val>
class LogReg : public Linear<Val> {
public:
  inline LogReg(const Args& args = {})
  : Linear<Val>{ LinearConf::AddObjectiveIfNotExists(
      args, LogisticLoss<Val>::NAME, 
      NegYLogLossMetric<Val, label_t, Val>::NAME) } {}
  inline const char* name() const override { return "LogisticRegression"; }
  inline bool use_neg_y() const override { return true; }
};

template <typename Val>
class SVM : public Linear<Val> {
public:
  inline SVM(const Args& args = {})
  : Linear<Val>{ LinearConf::AddObjectiveIfNotExists(
      args, HingeLoss<Val>::NAME, 
      HingeLossMetric<Val, label_t, Val>::NAME) } {}
  inline const char* name() const override { return "SupportVectorMachine"; }
  inline bool use_neg_y() const override { return true; }
};

template <typename Val>
class LinearReg : public Linear<Val> {
public:
  inline LinearReg(const Args& args = {})
  : Linear<Val>{ LinearConf::AddObjectiveIfNotExists(
      args, SquareLoss<Val>::NAME, 
      MSEMetric<Val, label_t, Val>::NAME) } {}
  inline const char* name() const override { return "LinearRegression"; }
  inline bool is_regression() const { return true; }
};

} // namespace linear
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LINEAR_LINEAR_H_
