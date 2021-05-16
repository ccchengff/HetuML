#ifndef __HETU_ML_MODEL_NAIVE_BAYES_PARALLEL_NAIVE_BAYES_H_
#define __HETU_ML_MODEL_NAIVE_BAYES_PARALLEL_NAIVE_BAYES_H_

#include "model/naive_bayes/naive_bayes.h"
#include "ps/psmodel/PSVector.h"

namespace hetu { 
namespace ml {
namespace naive_bayes {

template <typename Val>
class ParallelNaiveBayes : public NaiveBayes<Val> {
public:
  inline ParallelNaiveBayes(const Args& args = {}): NaiveBayes<Val>(args) {}

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

    NaiveBayes<Val>::InitModel(global_max_dim);
    auto num_label = this->params->num_label;
    this->ps_contigent_probability.reset(new PSVector<Val>(
      "contigent_probability.", num_label));
    this->ps_mean_vec.reset(new PSVector<Val>(
      "mean", global_max_dim * num_label));
    this->ps_var_vec.reset(new PSVector<Val>(
      "var", global_max_dim * num_label));
    if (rank == 0) {
      this->ps_contigent_probability->initAllZeros();
      this->ps_mean_vec->initAllZeros();
      this->ps_var_vec->initAllZeros();
    }
    PSAgent<Val>::Get()->barrier();
  }

  inline void ComputeStaticstics() override {
    this->ps_contigent_probability->densePush(
      this->contigent_probability.data(), 
      this->contigent_probability.size()
    );
    this->ps_mean_vec->densePush(
      this->mean_vec.data(), 
      this->mean_vec.size()
    );
    this->ps_var_vec->densePush(
      this->var_vec.data(), 
      this->var_vec.size()
    );
    PSAgent<Val>::Get()->barrier();
    this->ps_contigent_probability->densePull(
      this->contigent_probability.data(), 
      this->contigent_probability.size()
    );
    this->ps_mean_vec->densePull(
      this->mean_vec.data(), 
      this->mean_vec.size()
    );
    this->ps_var_vec->densePull(
      this->var_vec.data(), 
      this->var_vec.size()
    );
    NaiveBayes<Val>::ComputeStaticstics();
  }

  std::unique_ptr<PSVector<Val>> ps_contigent_probability;
  std::unique_ptr<PSVector<Val>> ps_mean_vec;
  std::unique_ptr<PSVector<Val>> ps_var_vec;
};

} // namespace naive_bayes
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_NAIVE_BAYES_PARALLEL_NAIVE_BAYES_H_
