#ifndef __HETU_ML_MODEL_GBDT_PARALLEL_GBDT_H_
#define __HETU_ML_MODEL_GBDT_PARALLEL_GBDT_H_

#include "model/gbdt/gbdt.h"
#include "model/gbdt/train/parallel_trainer.h"

namespace hetu { 
namespace ml {
namespace gbdt {

template <typename Val>
class ParallelGBDT : public GBDT<Val> {
public:
  inline ParallelGBDT(const Args& args = {}): GBDT<Val>(args) {}

  inline ~ParallelGBDT() {}

  inline void Fit(const Dataset<label_t, Val>& train_data, 
                  const Dataset<label_t, Val>& valid_data = {}) override {
    TIK(fit);
    HML_LOG_INFO << "Start to fit " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;

    for (size_t i = 0; i < 10; i++) {
      // 
    }
    
    GBDTDPTrainer trainer(*this->params);
    auto* model = trainer.Fit(train_data, valid_data);
    this->model.reset(model);
    
    TOK(fit);
    HML_LOG_INFO << "Fit " << this->name() << " model"
      << " cost " << COST_MSEC(fit) << " ms";
  }
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_PARALLEL_GBDT_H_
