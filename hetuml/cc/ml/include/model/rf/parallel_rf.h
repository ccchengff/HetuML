#ifndef __HETU_ML_MODEL_RF_PARALLEL_RF_H_
#define __HETU_ML_MODEL_RF_PARALLEL_RF_H_

#include "model/rf/rf.h"
#include "model/gbdt/parallel_gbdt.h"

namespace hetu { 
namespace ml {
namespace rf {

template <typename Val>
class ParallelRF : public gbdt::ParallelGBDT<Val> {
public:
  inline ParallelRF(const Args& args = {})
  : ParallelGBDT<Val>{ RFConf::AddRFArgs(args) } {}
  inline const char* name() const override { return "RandomForest"; }
};

} // namespace rf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_RF_PARALLEL_RF_H_
