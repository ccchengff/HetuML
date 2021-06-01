#ifndef __HETU_ML_MODEL_LDA_LDA_H_
#define __HETU_ML_MODEL_LDA_LDA_H_

#include "model/common/mlbase.h"
#include "model/lda/common/atomic_integer.h"
#include "model/lda/common/corpus.h"
#include "model/lda/common/ftree.h"
#include "model/lda/common/random.h"
#include "model/lda/common/sparse_counter.h"

namespace hetu { 
namespace ml {
namespace lda {

// utils for MH sampler
#define MH_STEP (2)
struct Token {
  int topic;
  std::vector<int> mh_step;
};

namespace LDAConf {
  // number of words
  static const std::string NUM_WORDS = "NUM_WORDS";
  // number of topics
  static const std::string NUM_TOPICS = "NUM_TOPICS";
  static const int DEFAULT_NUM_TOPICS = 100;
  // alpha
  static const std::string ALPHA = "ALPHA";
  static const float DEFAULT_ALPHA = 0;
  // beta
  static const std::string BETA = "BETA";
  static const float DEFAULT_BETA = 0.01;
  // number of iterations
  static const std::string NUM_ITERS = "NUM_ITERS";
  static const int DEFAULT_NUM_ITERS = 100;
  // threshold for long docs. The tradeoff used in Hybrid sampler.
  static const std::string LONG_DOC_THRES = "LONG_DOC_THRES";
  static const int DEFAULT_LONG_DOC_THRES = 600;

  static std::vector<std::string> meaningful_keys() {
    return {
      NUM_TOPICS, 
      NUM_WORDS, 
      ALPHA, 
      BETA, 
      NUM_ITERS, 
      LONG_DOC_THRES
    };
  }

  static Args default_args() { 
    return {
      { NUM_TOPICS, std::to_string(0) }, 
      { NUM_WORDS, std::to_string(DEFAULT_NUM_TOPICS) }, 
      { ALPHA, std::to_string(DEFAULT_ALPHA) }, 
      { BETA, std::to_string(DEFAULT_BETA) }, 
      { NUM_ITERS, std::to_string(DEFAULT_NUM_ITERS) }, 
      { LONG_DOC_THRES, std::to_string(DEFAULT_LONG_DOC_THRES) }
    };
  }

}; // LDAConf

class LDAParam : public MLParam {
public:
  LDAParam(const Args& args = {}, 
           const Args& default_args = LDAConf::default_args())
  : MLParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return LDAConf::meaningful_keys();
  }

  int32_t num_topics;
  int32_t num_words;
  float alpha;
  float beta;
  int32_t num_iters;
  int32_t long_doc_thres;

private:
  inline void InitAndCheckParam() override {
    this->num_topics = argparse::Get<int>(
      this->all_args, LDAConf::NUM_TOPICS);
    this->num_words = argparse::Get<int>(
      this->all_args, LDAConf::NUM_WORDS);
    this->alpha = argparse::Get<float>(
      this->all_args, LDAConf::ALPHA);
    if (this->alpha == 0) {
      alpha = 50.0 / num_topics;
    } else {
      ASSERT_GT(this->alpha, 0) 
        << "Invalid alpha: " << this->alpha;
    }
    this->beta = argparse::Get<float>(
      this->all_args, LDAConf::BETA);
    ASSERT_GT(this->beta, 0) 
      << "Invalid beta: " << this->beta;
    this->num_iters = argparse::Get<int>(
      this->all_args, LDAConf::NUM_ITERS);
    ASSERT_GT(this->num_iters, 0) 
      << "Invalid number of iterations: " << this->num_iters;
    this->long_doc_thres = argparse::Get<int>(
      this->all_args, LDAConf::LONG_DOC_THRES);
  }
};

/* Hybrid sampler for LDA. 
 * \cite{LDA*: A Robust and Large-scale Topic Modeling 
 * System, Lele Yu et al. VLDB2017}
 * for short docs, use F+LDA (SA samplers),
 * for long docs, use WarpLDA (MH sampler) 
 */
class LDA : public MLBase<LDAParam> {
public:
  inline LDA(const Args& args = {})
  : MLBase<LDAParam>(args) {}

  inline ~LDA() {}

  void Fit(const Corpus& corpus) {
    TIK(fit);
    HML_LOG_INFO << "Start to fit " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;
    this->PrepareForFit(corpus);
    for (int iter_id = 0; iter_id < this->params->num_iters; iter_id++) {
      TIK(iter);
      auto llh = this->SampleOneIteration(corpus, true);
      TOK(iter);
      HML_LOG_INFO << "Iteration[" << iter_id + 1 << "] loglikelihood[" 
        << std::fixed << llh << "] cost " << COST_MSEC(iter) << " ms";
    }
    TOK(fit);
    HML_LOG_INFO << "Fit " << this->name() << " model"
      << " cost " << COST_MSEC(fit) << " ms";
  }

  std::vector<int> Predict(const Corpus& corpus) {
    TIK(inference);
    this->PrepareForPredict(corpus);
    for (int iter_id = 0; iter_id < this->params->num_iters; iter_id++) {
      TIK(iter);
      auto llh = this->SampleOneIteration(corpus, false);
      TOK(iter);
      HML_LOG_INFO << "Iteration[" << iter_id + 1 << "] loglikelihood[" 
        << std::fixed << llh << "] cost " << COST_MSEC(iter) << " ms";
    }
    std::vector<int> ret(corpus.n_tokens);
    for (int i = 0; i < corpus.n_tokens; i++)
      ret[i] = topics[i].topic;
    TOK(inference);
    HML_LOG_INFO << "Inference cost " << COST_MSEC(inference) << " ms";
    return std::move(ret);
  }

  inline void LoadFromStream(std::istream& is) override {
    int n_topics = this->params->num_topics;
    int n_words = this->params->num_words;
    word_topic_dist.resize(n_words * n_topics);
    for (int nn = 0; nn < n_words * n_topics; nn++) {
      is >> word_topic_dist[nn];
    }
    topic_dist.resize(n_topics);
    for (int nt = 0; nt < n_topics; nt++) {
      int t;
      is >> t;
      topic_dist[nt].store(t);
    }
  }

  inline void DumpToStream(std::ostream& os) override {
    int n_topics = this->params->num_topics;
    int n_words = this->params->num_words;
    for (int nn = 0; nn < n_words * n_topics; nn++) {
      os << word_topic_dist[nn] << std::endl;
    }
    for (int nt = 0; nt < n_topics; nt++) {
      os << topic_dist[nt].load() << std::endl;
    }
  }

  inline const char* name() const override { return "LDA"; }

protected:
  /*
  * initialize topics(t.topic, t.mh_step), doc_topic_list for short docs,
  * topic_dist
  */
  virtual void PrepareForFit(const Corpus& corpus);

  virtual void PrepareForPredict(const Corpus& corpus);

  virtual double SampleOneIteration(const Corpus& corpus, bool update);

  // deal with short docs using F+ tree.
  // iterate over words in short docs.
  void FTreeIteration(const Corpus& corpus, bool update);

  // deal with long docs, re-assign topics for each token in long docs. 
  // using MH sampler. q_doc ~C_{dk} + \alpha
  void VisitByDoc(const Corpus& corpus, bool update);

  // deal with long docs
  void VisitByWord(const Corpus& corpus, bool update);

  double Loglikelihood(const Corpus& corpus);

  inline float CalWordTopic(int word_count, int topic, 
                            int n_words, float beta) const {
    return (word_count + beta) / (topic_dist[topic] + n_words * beta);
  }

  std::vector<int> word_topic_dist;
  std::vector<AtomicInt> topic_dist;
  std::vector<SparseCounter> doc_topic_dist;
  
  std::vector<Token> topics;
  std::vector<bool> is_long_doc;
  Random rand[100];
};

} // namespace lda
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LDA_LDA_H_
