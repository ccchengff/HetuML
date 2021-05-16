#ifndef __HETU_ML_MODEL_MF_PARALLEL_MF_H_
#define __HETU_ML_MODEL_MF_PARALLEL_MF_H_

#include "model/mf/mf.h"
#include "model/mf/common/model_average.h"

namespace hetu { 
namespace ml {
namespace mf {

class ParallelMF : public MF {
public:
  inline ParallelMF(const Args& args = {}): MF(args) {
    this->mf_param->nr_workers = NumWorkers();
  }

  inline void Fit(mf_problem const* train_data, 
                  mf_problem const* valid_data = nullptr) {
    TIK(fit);
    HML_LOG_INFO << "Start to fit " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;
    modelAverge = new PSModelAverage();
    if (valid_data != nullptr) {
      this->model.reset(mf_train_with_validation(
        train_data, valid_data, *mf_param));
    } else {
      this->model.reset(mf_train(train_data, *mf_param));
    }
    delete modelAverge;
    TOK(fit);
    HML_LOG_INFO << "Fit " << this->name() << " model"
      << " cost " << COST_MSEC(fit) << " ms";
  }
};

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_PARALLEL_MF_H_
