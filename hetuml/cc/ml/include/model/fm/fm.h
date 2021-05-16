#ifndef __HETU_ML_MODEL_FM_FM_H_
#define __HETU_ML_MODEL_FM_FM_H_

#include "model/common/mlbase.h"
#include "model/linear/linear.h"

namespace hetu { 
namespace ml {
namespace fm {

using namespace hetu::ml::linear;

namespace FMConf {
  static const std::string EMBEDDING_DIM = "EMBEDDING_DIM";
  static const int DEFAULT_EMBEDDING_DIM = 16;

  static std::vector<std::string> meaningful_keys() {
    const auto& linear_keys = LinearConf::meaningful_keys();
    std::vector<std::string> fm_keys = { EMBEDDING_DIM };
    fm_keys.insert(fm_keys.end(), linear_keys.begin(), linear_keys.end());
    return std::move(fm_keys);
  }

  static Args default_args() { 
    const auto& linear_args = LinearConf::default_args();
    Args fm_args = { {EMBEDDING_DIM, std::to_string(DEFAULT_EMBEDDING_DIM)} };
    fm_args.insert(linear_args.begin(), linear_args.end());
    fm_args[LinearConf::LOSS] = \
      LogisticLoss<label_t>::NAME;
    fm_args[LinearConf::METRICS] = \
      NegYLogLossMetric<label_t, label_t, label_t>::NAME;
    return std::move(fm_args);
  }
} // FMConf

class FMParam : public LinearParam {
public:
  FMParam(const Args& args = {}, 
          const Args& default_args = FMConf::default_args())
  : LinearParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return FMConf::meaningful_keys();
  }

  int embedding_dim;
private:
  inline void InitAndCheckParam() override {
    LinearParam::InitAndCheckParam();
    this->embedding_dim = argparse::Get<int>(
      this->all_args, FMConf::EMBEDDING_DIM);
    ASSERT_GT(embedding_dim, 0) 
      << "Illegal number of embedding dimensions: " << embedding_dim;
  }
};

template <typename Val>
class FM : public SupervisedMLBase<Val, FMParam> {
public:
  inline FM(const Args& args = {})
  : SupervisedMLBase<Val, FMParam>(args) {}

  inline ~FM() {}

  inline void Fit(const Dataset<label_t, Val>& train_data, 
                  const Dataset<label_t, Val>& valid_data = {}) override {
    ASSERT(!train_data.is_dense()) 
      << "Currently we only support sparse features for FM models";
    ASSERT(!valid_data.is_dense() || valid_data.get_num_instances() == 0)
      << "Currently we only support sparse features for FM models";
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
      HML_LOG_INFO << "Epoch[" << eid << "] Train loss[" << train_loss << "]";
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
      << "Currently we only support sparse features for FM models";
    // compute intermediate results
    std::vector<Val> inter;
    this->PredictInter(inter, features, start_id, end_id);
    // combine into predictions
    size_t batch_size = end_id - start_id;
    size_t embedding_dim = this->params->embedding_dim;
    ret.resize(batch_size);
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      auto offset = ins_id - start_id;
      // pred = <w, x> - 0.5 * sum_f(<v_f^2, x^2>) + 0.5 * sum(<v_f, x>^2)
      Val pred = inter[offset]; // <w, x> - 0.5 * sum_f(<v_f^2, x^2>)
      for (size_t embed_id = 0; embed_id < embedding_dim; embed_id++) {
        // <v_f, x>
        auto tmp = inter[(embed_id + 1) * batch_size + offset];
        pred += 0.5 * SQUARE(tmp);
      }
      ret[offset] = pred;
    }
  }

  inline void LoadFromStream(std::istream& is) override {
    size_t max_dim;
    is >> max_dim;
    this->InitModel(max_dim);
    for (auto& vec : this->model) {
      for (size_t dim = 0; dim < max_dim; dim++) {
        is >> vec->values[dim];
      }
    }
  }

  inline void DumpToStream(std::ostream& os) override {
    size_t max_dim = this->get_max_dim();
    os << max_dim << std::endl;
    for (auto& vec : this->model) {
      for (size_t dim = 0; dim < max_dim; dim++) {
        os << vec->values[dim] << std::endl;
      }
    }
  }

  inline bool is_empty() const { 
    return this->model.empty(); 
  }

  inline size_t get_max_dim() const {
    ASSERT(!this->is_empty()) << "Model is empty";
    return this->model[0]->dim;
  }

  inline bool use_neg_y() const override { return true; }

  inline const char* name() const override { return "FactorizationMachine"; }

protected:
  virtual void InitModel(size_t max_dim) {
    size_t embedding_dim = this->params->embedding_dim;
    ASSERT_GT(max_dim, 0) 
      << "Illegal number of dimensions: " << max_dim;
    ASSERT_GT(embedding_dim, 0) 
      << "Illegal number of embedding dimensions: " << embedding_dim;
    this->model.resize(embedding_dim + 1);
    for (size_t i = 0; i <= embedding_dim; i++) {
      this->model[i].reset(new DenseVector<Val>(max_dim));
      // XavierNormal init
      NormalDistribution<Val>(this->model[i]->values, max_dim, 
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
    // compute intermediate results
    std::vector<Val> inter;
    this->PredictInter(inter, train_data, start_id, end_id);
    // compute loss, grad, and update model
    Val batch_loss = 0;
    const auto* loss_func = LossFactory::GetLoss<Val>(this->params->loss);
    size_t batch_size = end_id - start_id;
    size_t embedding_dim = this->params->embedding_dim;
    auto step = -this->params->learning_rate / batch_size;
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      const auto& label = train_data.get_label(ins_id);
      const auto& feature = train_data.get_sparse_feature(ins_id);
      // combine into prediction
      auto offset = ins_id - start_id;
      // pred = <w, x> - 0.5 * sum_f(<v_f^2, x^2>) + 0.5 * sum(<v_f, x>^2)
      Val pred = inter[offset]; // <w, x> - 0.5 * sum_f(<v_f^2, x^2>)
      for (size_t embed_id = 0; embed_id < embedding_dim; embed_id++) {
        // <v_f, x>
        auto embed_dot = inter[(embed_id + 1) * batch_size + offset];
        pred += 0.5 * SQUARE(embed_dot);
      }
      // compute loss
      batch_loss += loss_func->loss(pred, label);
      Val grad_multipler = loss_func->grad(pred, label);
      // update w
      this->model[0]->axp0(feature, step * grad_multipler);
      // update v_f
      for (size_t embed_id = 0; embed_id < embedding_dim; embed_id++) {
        auto& embed = this->model[embed_id + 1];
        auto embed_dot = inter[(embed_id + 1) * batch_size + offset];
        for (size_t i = 0; i < feature.nnz; i++) {
          auto indice = feature.indices[i];
          auto value = feature.values[i];
          embed->values[indice] += step * grad_multipler * value * \
            (embed_dot - embed->values[indice] * value);
        }
      }
    }
    return batch_loss / (end_id - start_id);
  }

  inline void PredictInter(std::vector<label_t>& ret, 
                           const DataMatrix<Val>& features, 
                           size_t start_id, size_t end_id) {
    auto embedding_dim = this->params->embedding_dim;
    size_t batch_size = end_id - start_id;
    // size of ret: [(#embedding + 1), batch_size]
    // for each sample, we store:
    // (1) <w, x> - 0.5 * sum_f(<v_f^2, x^2>)
    // (2) <v_f, x> (for each embedding)
    ret.resize((embedding_dim + 1) * batch_size);
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      auto offset = ins_id - start_id;
      const auto& feature = features.get_sparse_feature(ins_id);
      Val linear_dot = this->model[0]->dot(feature);
      ret[offset] = linear_dot;
      for (size_t embed_id = 0; embed_id < embedding_dim; embed_id++) {
        Val m1 = this->model[embed_id + 1]->dot(feature);
        Val m2 = this->model[embed_id + 1]->dot_square(feature);
        ret[(embed_id + 1) * batch_size + offset] = m1;
        ret[offset] -= 0.5 * m2;
      }
    }
  }

  // first vector: linear (w); other vectors: embedding (v_f)
  std::vector<std::unique_ptr<DenseVector<Val>>> model;
};

} // namespace fm
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_FM_FM_H_
