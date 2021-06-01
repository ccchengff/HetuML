#ifndef __HETU_ML_MODEL_NAIVE_BAYES_NAIVE_BAYES_H_
#define __HETU_ML_MODEL_NAIVE_BAYES_NAIVE_BAYES_H_

#include "model/common/mlbase.h"
#include "common/threading.h"

namespace hetu { 
namespace ml {
namespace naive_bayes {

namespace NaiveBayesConf {
  // number of labels
  static const std::string NUM_LABEL = "NUM_LABEL";
  static const int DEFAULT_NUM_LABEL = 2;
  // evaluation metric
  static const std::string METRICS = "METRICS";

  static std::vector<std::string> meaningful_keys() {
    return {
      NUM_LABEL, 
      METRICS
    };
  }

  static Args default_args() { 
    return {
      { NUM_LABEL, std::to_string(DEFAULT_NUM_LABEL) }, 
      { METRICS, "" }
    };
  }
} // NaiveBayesConf

class NaiveBayesParam : public MLParam {
public:
  NaiveBayesParam(const Args& args = {}, 
                  const Args& default_args = NaiveBayesConf::default_args())
  : MLParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return NaiveBayesConf::meaningful_keys();
  }

  int num_label;
  std::vector<std::string> metrics;
private:
  inline void InitAndCheckParam() override {
    this->num_label = argparse::Get<int>(
      this->all_args, NaiveBayesConf::NUM_LABEL);
    ASSERT_GT(this->num_label, 0) 
      << "Invalid number of labels: " << this->num_label;
    this->metrics = argparse::GetVector<std::string>(
      this->all_args, NaiveBayesConf::METRICS);
    // check whether metrics are properly set
    for (const auto& metric : this->metrics) {
      const auto* eval_metric = \
        MetricFactory::GetEvalMetric<label_t, label_t, label_t>(metric, false);
      ASSERT(eval_metric != nullptr) << "Undefined metric: " << metric;
    }
  }
};

template <typename Val>
class NaiveBayes : public SupervisedMLBase<Val, NaiveBayesParam> {
public:
  inline NaiveBayes(const Args& args = {})
  : SupervisedMLBase<Val, NaiveBayesParam>(args) {}

  inline ~NaiveBayes() {}

  inline void Fit(const Dataset<label_t, Val>& train_data, 
                  const Dataset<label_t, Val>& valid_data = {}) override {
    ASSERT(!train_data.is_dense()) 
      << "Currently we only support sparse features for Linear models";
    ASSERT(!valid_data.is_dense() || valid_data.get_num_instances() == 0)
      << "Currently we only support sparse features for Linear models";
    this->InitModel(train_data.get_max_dim());

    TIK(fit);
    HML_LOG_INFO << "Start to fit " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;
    
    auto max_dim = this->get_max_dim();
    auto num_label = this->params->num_label;
    auto num_train = train_data.get_num_instances();
    auto num_valid = valid_data.get_num_instances();
    const auto& metrics = this->params->metrics;

    this->SummarizeDataset(train_data);
    this->ComputeStaticstics();

    // evaluation
    if (!metrics.empty()) {
      auto train_metrics = this->Evaluate(train_data, metrics);
      HML_LOG_INFO << "Evaluation on train data: " 
        << MetricFactory::to_string(metrics, train_metrics);
      if (num_valid > 0) {
        auto valid_metrics = this->Evaluate(valid_data, metrics);
        HML_LOG_INFO << "Evaluation on valid data: " 
          << MetricFactory::to_string(metrics, valid_metrics);
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
      << "Currently we only support sparse features for NaiveBayes models";
    
    auto max_dim = this->get_max_dim();
    auto num_label = this->params->num_label;
    ret.resize((end_id - start_id) * num_label);
    #pragma omp parallel for schedule(dynamic)
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      label_t* prob = ret.data() + (ins_id - start_id) * num_label;
      std::fill(prob, prob + num_label, (Val) 1);
      const auto& feature = features.get_sparse_feature(ins_id);
      for (size_t i = 0; i < feature.nnz; i++) {
        auto indices = feature.indices[i];
        auto value = feature.values[i];
        if (indices < max_dim) {
          for (size_t k = 0; k < num_label; k++) {
            auto mean = this->mean_vec[indices * num_label + k];
            auto var = this->var_vec[indices * num_label + k];
            if (var < EPSILON) continue;
            auto pred = this->Normal(value, mean, var);
            prob[k] *= MAX(pred, EPSILON);
          }
        }
      }
      for (size_t k = 0; k < num_label; k++) {
        prob[k] *= this->contigent_probability[k];
      }
    }
  }

  inline void LoadFromStream(std::istream& is) override {
    size_t max_dim;
    is >> max_dim;
    this->InitModel(max_dim);
    for (size_t i = 0; i < this->params->num_label; i++) 
      is >> this->contigent_probability[i];
    for (size_t i = 0; i < max_dim * this->params->num_label; i++)
      is >> this->mean_vec[i];
    for (size_t i = 0; i < max_dim * this->params->num_label; i++)
      is >> this->var_vec[i];
  }

  inline void DumpToStream(std::ostream& os) override {
    size_t max_dim = this->get_max_dim();
    os << max_dim << std::endl;
    for (Val v : this->contigent_probability) 
      os << v << std::endl;
    for (Val v : this->mean_vec) 
      os << v << std::endl;
    for (Val v : this->var_vec) 
      os << v << std::endl;
  }

  inline const char* name() const override { return "NaiveBayes"; }

  inline bool is_empty() const { 
    return this->contigent_probability.empty() || 
      this->mean_vec.empty() || this->var_vec.empty();
  }

  inline size_t get_max_dim() const {
    ASSERT(!this->is_empty()) << "Model is empty";
    return this->mean_vec.size() / this->params->num_label;
  }

  inline int get_num_label() const override { return this->params->num_label; }

protected:
  virtual void InitModel(size_t max_dim) {
    auto num_label = this->params->num_label;
    contigent_probability.resize(num_label, 0);
    mean_vec.resize(max_dim * num_label, 0);
    var_vec.resize(max_dim * num_label, 0);
  }

  inline void SummarizeDataset(const Dataset<label_t, Val>& dataset) {
    auto max_dim = this->get_max_dim();
    auto num_label = this->params->num_label;
    auto num_ins = dataset.get_num_instances();

    // use double to maintain precision
    std::vector<double> contigent_probability_buffer(num_label, 0);
    std::vector<double> mean_vec_buffer(max_dim * num_label, 0);
    std::vector<double> var_vec_buffer(max_dim * num_label, 0);

    #pragma omp parallel for schedule(dynamic) \
      reduction(vec_double_plus:contigent_probability_buffer, \
        mean_vec_buffer, var_vec_buffer)
    for (size_t ins_id = 0; ins_id < num_ins; ins_id++) {
      const auto label = static_cast<int>(dataset.get_label(ins_id));
      const auto& feature = dataset.get_sparse_feature(ins_id);
      ASSERT(label >= 0 && label < num_label) 
        << "Invalid label: " << label 
        << ", expected in [0, " << num_label << ")";
      contigent_probability_buffer[label]++;
      for (size_t i = 0; i < feature.nnz; i++) {
        auto indices = feature.indices[i];
        auto value = feature.values[i];
        mean_vec_buffer[indices * num_label + label] += value;
        var_vec_buffer[indices * num_label + label] += SQUARE(value);
      }
    }

    for (size_t i = 0; i < num_label; i++) {
      this->contigent_probability[i] += contigent_probability_buffer[i];
    }
    for (size_t i = 0; i < max_dim * num_label; i++) {
      this->mean_vec[i] += mean_vec_buffer[i];
      this->var_vec[i] += var_vec_buffer[i];
    }
  }

  inline virtual void ComputeStaticstics() {
    auto max_dim = this->get_max_dim();
    auto num_label = this->params->num_label;
    #pragma omp parallel for
    for (size_t dim = 0; dim < max_dim; dim++) {
      for (size_t k = 0; k < num_label; k++) {
        auto index = dim * num_label + k;
        auto label_cnt = this->contigent_probability[k];
        if (label_cnt == 0) continue; // no instances have such a label
        this->mean_vec[index] /= label_cnt;
        this->var_vec[index] /= label_cnt;
        this->var_vec[index] -= SQUARE(this->mean_vec[index]);
      }
    }
    Val num_ins = std::accumulate(this->contigent_probability.begin(), 
      this->contigent_probability.end(), (Val) 0);
    #pragma omp parallel for
    for (size_t k = 0; k < num_label; k++) {
      this->contigent_probability[k] /= num_ins;
    }
  }

  inline Val Normal(Val x, Val mean, Val var) const {
    return Exp(-0.5 * SQUARE(x - mean) / var) / std::sqrt(2 * PI * var);
  }

  std::vector<Val> contigent_probability;
  std::vector<Val> mean_vec;
  std::vector<Val> var_vec;
};

} // namespace naive_bayes
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_NAIVE_BAYES_NAIVE_BAYES_H_
