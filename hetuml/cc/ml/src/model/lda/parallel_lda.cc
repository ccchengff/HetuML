#include "model/lda/parallel_lda.h"

namespace hetu { 
namespace ml {
namespace lda {

void ParallelLDA::PrepareForFit(const Corpus &corpus) {
  LDA::PrepareForFit(corpus);

  // init topic_dist, word_topic_dist on PS
  int n_topics = this->params->num_topics;
  int n_words = this->params->num_words;
  ps_topic_dist.reset(new PSVector<int>("topic_dist", n_topics));
  ps_word_topic_dist.reset(new PSVector<int>(
    "word_topic_dist", n_words * n_topics));
  int rank = MyRank();
  if (rank == 0) {
    ps_topic_dist->initAllZeros();
    ps_word_topic_dist->initAllZeros();
  }
  PSAgent<int>::Get()->barrier();

  // init local buffer
  topic_dist_buffer.resize(n_topics);
  word_topic_dist_buffer.resize(n_topics * n_words);

  // push topic_dist, word_topic_dist to PS
  for (int nt = 0; nt < n_topics; nt++) {
    topic_dist_buffer[nt] = topic_dist[nt].load(std::memory_order_seq_cst);
  }
  ps_topic_dist->densePush(topic_dist_buffer.data(), n_topics);
  ps_word_topic_dist->densePush(word_topic_dist.data(), n_topics * n_words);
  PSAgent<int>::Get()->barrier();
}

float ParallelLDA::SampleOneIteration(const Corpus& corpus, bool update) {
  int n_topics = this->params->num_topics;
  int n_words = this->params->num_words;

  // pull topic_dist and word_topic_dist from PS
  if (update) {
    ps_topic_dist->densePull(topic_dist_buffer.data(), n_topics);
    #pragma omp parallel for schedule(dynamic)
    for (int nt = 0; nt < n_topics; nt++) {
      topic_dist[nt].store(topic_dist_buffer[nt]);
    }
    ps_word_topic_dist->densePull(word_topic_dist_buffer.data(), 
      n_topics * n_words);
    std::copy(word_topic_dist_buffer.begin(), word_topic_dist_buffer.end(), 
      word_topic_dist.begin());
  }

  // fit one iteration on local data shard
  float llh = LDA::SampleOneIteration(corpus, update);

  // push topic_dist and word_topic_dist to PS
  if (update) {
    #pragma omp parallel for schedule(dynamic)
    for (int nt = 0; nt < n_topics; nt++) {
      topic_dist_buffer[nt] = topic_dist[nt] - topic_dist_buffer[nt];
    }
    ps_topic_dist->densePush(topic_dist_buffer.data(), n_topics);
    int nnz = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+ : nnz)
    for (int nn = 0; nn < n_words * n_topics; nn++) {
      word_topic_dist_buffer[nn] = \
        word_topic_dist[nn] - word_topic_dist_buffer[nn];
      if (word_topic_dist_buffer[nn] != 0) nnz++;
    }
    if (nnz < n_words * n_topics / 2) { // sparse
      indices_buffer.clear(); indices_buffer.reserve(nnz);
      values_buffer.clear(); values_buffer.reserve(nnz);
      for (int nn = 0; nn < n_words * n_topics; nn++) {
        if (word_topic_dist_buffer[nn] != 0) {
          indices_buffer.push_back(nn);
          values_buffer.push_back(word_topic_dist_buffer[nn]);
          if (indices_buffer.size() == nnz) break;
        }
      }
      ps_word_topic_dist->sparsePush(indices_buffer.data(), 
        values_buffer.data(), nnz, true);
    } else { // dense
      ps_word_topic_dist->densePush(word_topic_dist_buffer.data(), 
        n_topics * n_words);
    }
  }

  return llh;
}

} // namespace lda
} // namespace ml
} // namespace hetu
