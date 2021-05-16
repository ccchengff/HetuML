#ifndef __HETU_ML_MODEL_LINEAR_PARALLEL_LINEAR_H_
#define __HETU_ML_MODEL_LINEAR_PARALLEL_LINEAR_H_

#include "model/linear/linear.h"
#include "ps/psmodel/PSVector.h"

namespace hetu { 
namespace ml {
namespace linear {

template <typename Val>
class ParallelLinear : public Linear<Val> {
public:
  inline ParallelLinear(const Args& args = {}): Linear<Val>(args) {}

private:
  inline void InitModel(size_t max_dim) override {
    // sync max dim in order to avoid data skewness
    int rank = MyRank();
    int num_workers = NumWorkers();
    std::vector<Val> max_dims(num_workers);
    PSVector<Val> ps_max_dims("max_dims", num_workers);
    if (rank == 0) ps_max_dims.initAllZeros();
    PSAgent<Val>::Get()->barrier();
    auto local_max_dim = static_cast<Val>(max_dim);
    ps_max_dims.sparsePush(&rank, &local_max_dim, 1, false);
    PSAgent<Val>::Get()->barrier();
    ps_max_dims.densePull(max_dims.data(), num_workers);
    auto global_max_dim = static_cast<size_t>(*std::max_element(
      max_dims.begin(), max_dims.end()));
    
    // init local and ps model with global max dim
    Linear<Val>::InitModel(global_max_dim);
    this->ps_model.reset(new PSVector<Val>(
      "linear", global_max_dim));
    if (rank == 0) {
      this->ps_model->initAllZeros();
      this->ps_model->densePush(this->model->values, global_max_dim);
    }
    PSAgent<Val>::Get()->barrier();
    this->bitmap.resize(global_max_dim);
    this->values_buffer.reset(new DenseVector<Val>(max_dim));
  }

  inline Val FitOneEpoch(const Dataset<label_t, Val>& train_data) override {
    auto loss = Linear<Val>::FitOneEpoch(train_data);
    // pull entire model from PS after each epoch
    this->ps_model->densePull(this->model->values, this->model->dim);
    return loss;
  }

  Val FitOneBatch(const Dataset<label_t, Val>& train_data, 
                  size_t start_id, size_t end_id) override {
    // pull model from PS
    this->SparsePullModel(train_data, start_id, end_id);
    
    // mini-batch
    Val batch_loss = 0;
    const auto* loss_func = LossFactory::GetLoss<Val>(this->params->loss);
    // accumulate gradients in buffer
    this->values_buffer->clear();
    auto tmp = -this->params->learning_rate / (end_id - start_id);
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      const auto& label = train_data.get_label(ins_id);
      const auto& feature = train_data.get_sparse_feature(ins_id);
      Val dot = this->model->dot(feature);
      batch_loss += loss_func->loss(dot, label);
      Val grad_multipler = loss_func->grad(dot, label);
      this->values_buffer->axp0(feature, grad_multipler * tmp);
    }
    
    // update model on PS
    this->SparsePushModel();
    PSAgent<Val>::Get()->barrier();
    return batch_loss / (end_id - start_id);
  }

  inline void SparsePullModel(const Dataset<label_t, Val>& train_data, 
                              size_t start_id, size_t end_id) {
    // check the appearance of each dimension
    std::fill(this->bitmap.begin(), this->bitmap.end(), false);
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      const auto& feature = train_data.get_sparse_feature(ins_id);
      for (size_t i = 0; i < feature.nnz; i++) {
        this->bitmap[feature.indices[i]] = true;
      }
    }
    // sparse pull from PS
    this->indices_buffer.clear();
    for (size_t dim = 0; dim < this->bitmap.size(); dim++) {
      if (this->bitmap[dim]) 
        this->indices_buffer.push_back(dim);
    }
    this->ps_model->sparsePull(this->indices_buffer.data(), 
      this->values_buffer->values, this->indices_buffer.size(), false);
    // update local model
    for (size_t i = 0; i < this->indices_buffer.size(); i++) {
      this->model->values[this->indices_buffer[i]] = \
        this->values_buffer->values[i];
    }
  }

  inline void SparsePushModel() {
    // extract updates for appeared dimensions
    for (size_t i = 0; i < this->indices_buffer.size(); i++) {
      auto dim = this->indices_buffer[i];
      this->values_buffer->values[i] = this->values_buffer->values[dim];
    }
    // sparse push to PS
    this->ps_model->sparsePush(this->indices_buffer.data(), 
      this->values_buffer->values, this->indices_buffer.size(), true);
  }

  std::unique_ptr<PSVector<Val>> ps_model;
  
  // some workspace during fitting
  std::vector<bool> bitmap;
  std::vector<int> indices_buffer;
  std::unique_ptr<DenseVector<Val>> values_buffer;
};

template <typename Val>
class ParallelLogReg : public ParallelLinear<Val> {
public:
  inline ParallelLogReg(const Args& args = {})
  : ParallelLinear<Val>{ LinearConf::AddObjectiveIfNotExists(
      args, LogisticLoss<Val>::NAME, 
      NegYLogLossMetric<Val, label_t, Val>::NAME) } {}
  inline const char* name() const override { return "LogisticRegression"; }
};

template <typename Val>
class ParallelSVM : public ParallelLinear<Val> {
public:
  inline ParallelSVM(const Args& args = {})
  : ParallelLinear<Val>{ LinearConf::AddObjectiveIfNotExists(
      args, HingeLoss<Val>::NAME, 
      HingeLossMetric<Val, label_t, Val>::NAME) } {}
  inline const char* name() const override { return "SupportVectorMachine"; }
};

template <typename Val>
class ParallelLinearReg : public ParallelLinear<Val> {
public:
  inline ParallelLinearReg(const Args& args = {})
  : ParallelLinear<Val>{ LinearConf::AddObjectiveIfNotExists(
      args, SquareLoss<Val>::NAME, 
      RMSEMetric<Val, label_t, Val>::NAME) } {}
   inline const char* name() const override { return "LinearRegression"; }
};


} // namespace linear
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LINEAR_PARALLEL_LINEAR_H_
