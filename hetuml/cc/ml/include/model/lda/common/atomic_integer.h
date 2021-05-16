#ifndef __HETU_ML_MODEL_LDA_COMMON_ATOMIC_INT_H_
#define __HETU_ML_MODEL_LDA_COMMON_ATOMIC_INT_H_

#include <atomic>

namespace hetu { 
namespace ml {
namespace lda {

struct AtomicInt : std::atomic_int {
  AtomicInt() : std::atomic_int(0) {}
  AtomicInt(AtomicInt &&b) { this->store(b); }
};

} // namespace lda
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LDA_COMMON_ATOMIC_INT_H_
