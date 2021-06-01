#ifndef __HETU_ML_MODEL_LDA_PARALLEL_LDA_H_
#define __HETU_ML_MODEL_LDA_PARALLEL_LDA_H_

#include "model/lda/lda.h"
#include "ps/psmodel/PSVector.h"
#include <memory>

namespace hetu { 
namespace ml {
namespace lda {

class ParallelLDA : public LDA {
public:
  inline ParallelLDA(const Args& args = {}): LDA(args) {}

  inline ~ParallelLDA() {}

private:
  void PrepareForFit(const Corpus& corpus) override;

  double SampleOneIteration(const Corpus& corpus, bool update) override;

  std::shared_ptr<PSVector<int>> ps_topic_dist;
  std::shared_ptr<PSVector<int>> ps_word_topic_dist;
  std::vector<int> topic_dist_buffer;
  std::vector<int> word_topic_dist_buffer;
  std::vector<int> indices_buffer;
  std::vector<int> values_buffer;
};

} // namespace lda
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LDA_PARALLEL_LDA_H_
