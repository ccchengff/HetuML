#ifndef __HETU_ML_MODEL_MF_COMMON_MODEL_AVERAGE_H_
#define __HETU_ML_MODEL_MF_COMMON_MODEL_AVERAGE_H_

#include "model/mf/common/util.h"
#include "ps/psmodel/PSVector.h"

namespace hetu { 
namespace ml {
namespace mf {

class PSModelAverage : public ModelAverageBase {
public:
  virtual void Init(mf_model& model) {
    this->p_size = (long) model.m * model.k;
    this->q_size = (long) model.n * model.k;
    this->rank = MyRank();
    this->num_workers = NumWorkers();
    this->ps_model = new PSVector<float>("mf_model", p_size + q_size);
    this->ps_buffer = Utility::malloc_aligned_float(p_size + q_size);
  }

  virtual void UpdateModel(mf_model& model) {
    if (rank == 0) ps_model->initAllZeros();
    
    for (long i = 0; i < p_size; i++) {
      if (!std::isnan(model.P[i]))
        ps_buffer[i] = model.P[i];
      else
        ps_buffer[i] = 0;
    }
    for (long i = 0; i < q_size; i++) {
      if (!std::isnan(model.Q[i]))
        ps_buffer[p_size + i] = model.Q[i];
      else
        ps_buffer[p_size + i] = 0;
    }

    PSAgent<float>::Get()->barrier();
    ps_model->densePush(ps_buffer, (int) (p_size + q_size));
    PSAgent<float>::Get()->barrier();
    ps_model->densePull(ps_buffer, (int) (p_size + q_size));

    for (long i = 0; i < p_size; i++) {
      if (!std::isnan(model.P[i]))
        model.P[i] = ps_buffer[i] / num_workers;
    }
    for (long i = 0; i < q_size; i++) {
      if (!std::isnan(model.Q[i]))
        model.Q[i] = ps_buffer[p_size + i] / num_workers;
    }
  }

  virtual void Finalize(mf_model& model) {
    delete this->ps_model;
    Utility::free_aligned_float(this->ps_buffer);
  }

  PSVector<float> *ps_model = nullptr;
  float* ps_buffer = nullptr;
  int rank; 
  int num_workers;
  long p_size;
  long q_size;
};

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_COMMON_MODEL_AVERAGE_H_
