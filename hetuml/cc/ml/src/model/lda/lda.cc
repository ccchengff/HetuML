#include "model/lda/lda.h"
#include "common/threading.h"

namespace hetu { 
namespace ml {
namespace lda {

void LDA::PrepareForFit(const Corpus &corpus) {
  int n_topics = this->params->num_topics;
  int n_words = this->params->num_words;
  ASSERT_GT(n_topics, 0) << "Invalid number of topics: " << n_topics;
  ASSERT_GT(n_words, 0) << "Invalid number of words: " << n_words;
  word_topic_dist.resize(n_words * n_topics);
  topics.resize(corpus.n_tokens);
  topic_dist.resize(n_topics);
  is_long_doc.resize(corpus.n_docs);
  doc_topic_dist.clear();

  for (int d = 0; d < corpus.n_docs; ++d) {
    is_long_doc[d] = (corpus.GetDocSize(d) > this->params->long_doc_thres);
    if (!is_long_doc[d]) {
      doc_topic_dist.emplace_back(n_topics);
    } else {
      doc_topic_dist.emplace_back(0);
    }
  }
  for (int w = 0; w < corpus.n_words; ++w) {
    int start = corpus.word_offset[w];
    int end = corpus.word_offset[w + 1];
    for (int i = start; i < end; ++i) {
      int doc = corpus.words[i];
      int topic = 0;
      topic = topics[i].topic = rand[0].RandInt(n_topics);
      if (is_long_doc[doc]) {
        for (int j = 0; j < MH_STEP; ++j) {
          topics[i].mh_step.push_back(rand[0].RandInt(n_topics));
        }
      }
      topic_dist[topic]++;
      word_topic_dist[w * n_topics + topic]++;
      if (!is_long_doc[doc]) {
        doc_topic_dist[doc].Inc(topic);
      }
    }
  }
}

void LDA::PrepareForPredict(const Corpus &corpus) {
  int n_topics = this->params->num_topics;
  int n_words = this->params->num_words;
  ASSERT_GT(n_topics, 0) << "Invalid number of topics: " << n_topics;
  ASSERT_GT(n_words, 0) << "Invalid number of words: " << n_words;
  topics.resize(corpus.n_tokens);
  is_long_doc.resize(corpus.n_docs);
  doc_topic_dist.clear();

  for (int d = 0; d < corpus.n_docs; ++d) {
    is_long_doc[d] = (corpus.GetDocSize(d) > this->params->long_doc_thres);
    if (!is_long_doc[d]) {
      doc_topic_dist.emplace_back(n_topics);
    } else {
      doc_topic_dist.emplace_back(0);
    }
  }
  for (int w = 0; w < corpus.n_words; ++w) {
    int start = corpus.word_offset[w];
    int end = corpus.word_offset[w + 1];
    for (int i = start; i < end; ++i) {
      int doc = corpus.words[i];
      int topic = 0;
      topic = topics[i].topic = rand[0].RandInt(n_topics);
      if (is_long_doc[doc]) {
        for (int j = 0; j < MH_STEP; ++j) {
          topics[i].mh_step.push_back(rand[0].RandInt(n_topics));
        }
      }
      if (!is_long_doc[doc]) {
        doc_topic_dist[doc].Inc(topic);
      }
    }
  }
}

double LDA::SampleOneIteration(const Corpus& corpus, bool update) {
  // long docs by MH, WARPLDA
  this->VisitByDoc(corpus, update);
  this->VisitByWord(corpus, update);
  
  // short docs by SA, F+LDA
  this->FTreeIteration(corpus, update);
  
  // calculate likelihood
  auto llh = this->Loglikelihood(corpus);
  
  return llh;
}

void LDA::FTreeIteration(const Corpus &corpus, bool update) {
  int n_topics = this->params->num_topics;
  float alpha = this->params->alpha;
  float beta = this->params->beta;
  FTree tree(n_topics);
  std::vector<float> psum(n_topics);

  #pragma omp parallel for schedule(dynamic) \
    firstprivate(psum, tree)
  for (int word = 0; word < corpus.n_words; ++word) {
    int thread = OMP_GET_THREAD_ID();
    for (int i = 0; i < n_topics; ++i) {
      tree.Set(i, CalWordTopic(word_topic_dist[word * n_topics + i], 
        i, corpus.n_words, beta));
    }
    tree.Build();
    int begin = corpus.word_offset[word];
    int end = corpus.word_offset[word + 1];
    for (int i = begin; i < end; ++i) {
      int doc = corpus.words[i];
      if (is_long_doc[doc])
        continue;
      
      int old_topic = topics[i].topic;
      
      if (update) {
        word_topic_dist[word * n_topics + old_topic]--;
        topic_dist[old_topic]--;
        auto wt = CalWordTopic(
          word_topic_dist[word * n_topics + old_topic], 
          old_topic, corpus.n_words, beta);
        tree.Update(old_topic, wt);
      }

      SparseCounter &doc_dist = doc_topic_dist[doc];
      doc_dist.Lock();
      doc_dist.Dec(old_topic);

      float prob_left = tree.Sum() * alpha;
      float prob_all = prob_left;
      const std::vector<CountItem> &items = doc_dist.GetItem();
      for (int t = 0; t < (int) items.size(); t++) {
        float p = items[t].count * tree.Get(items[t].item);
        prob_all += p;
        psum[t] = p;
        if (t > 0)
          psum[t] += psum[t - 1];
      }

      float prob = rand[thread].RandDouble(prob_all);
      int new_topic;
      if (prob < prob_left) {
        prob = rand[thread].RandDouble(prob_left);
        new_topic = tree.Sample(prob / alpha);
      } else {
        prob -= prob_left;
        prob = rand[thread].RandDouble(prob_all - prob_left);
        int p = std::lower_bound(psum.begin(), psum.begin() + items.size(), 
          prob) - psum.begin();
        new_topic = items[p].item;
      }

      doc_dist.Inc(new_topic);
      doc_dist.Unlock();
      
      if (update) {
        word_topic_dist[word * n_topics + new_topic]++;
        topic_dist[new_topic]++;
        auto new_wt = CalWordTopic(
          word_topic_dist[word * n_topics + new_topic], 
          new_topic, corpus.n_docs, beta);
        tree.Update(new_topic, new_wt);
      }
      
      topics[i].topic = new_topic;
    }
  }
}

void LDA::VisitByWord(const Corpus &corpus, bool update) {
  int n_topics = this->params->num_topics;
  float beta = this->params->beta;
  
  #pragma omp parallel for schedule(dynamic)
  for (int word = 0; word < corpus.n_words; ++word) {
    int thread = OMP_GET_THREAD_ID();
    int N = corpus.word_offset[word + 1] - corpus.word_offset[word];
    int offset = corpus.word_offset[word];
    if (update) {
      std::fill(word_topic_dist.begin() + word * n_topics, 
        word_topic_dist.begin() + (word + 1) * n_topics, 0);

      for (int i = 0; i < N; ++i) {
        word_topic_dist[word * n_topics + topics[offset + i].topic]++;
      }
    }

    for (int i = 0; i < N; ++i) {
      if (!is_long_doc[corpus.words[offset + i]])
        continue;
      int topic = topics[offset + i].topic;
      if (update) {
        word_topic_dist[word * n_topics + topic]--;
        topic_dist[topic]--;
      }

      for (int m = 0; m < MH_STEP; ++m) {
        int new_topic = topics[offset + i].mh_step[m];
        float Cwj = word_topic_dist[word * n_topics + new_topic] + beta;
        float Cwi = word_topic_dist[word * n_topics + topic] + beta;
        float Cj = topic_dist[new_topic] + corpus.n_words * beta;
        float Ci = topic_dist[topic] + corpus.n_words * beta;
        float prob = Cwj * Ci * rand[thread].MAX_N;
        topic = rand[thread].RandInt() * Cwi * Cj < prob ? new_topic : topic;
      }
      if (update) {
        topic_dist[topic]++;
        word_topic_dist[word * n_topics + topic]++;
      }
      topics[offset + i].topic = topic;
    }

    unsigned prob =
        (n_topics * beta) / (n_topics * beta + N) * rand[thread].MAX_N;
    for (int i = 0; i < N; ++i) {
      if (!is_long_doc[corpus.words[offset + i]])
        continue;
      for (int m = 0; m < MH_STEP; ++m) {
        int a = rand[thread].RandInt(n_topics);
        int b = topics[offset + rand[thread].RandInt(N)].topic;
        topics[offset + i].mh_step[m] =
          rand[thread].RandDouble() < prob ? a : b;
      }
    }
  }
}

void LDA::VisitByDoc(const Corpus &corpus, bool update) {
  int n_topics = this->params->num_topics;
  float alpha = this->params->alpha;
  float beta = this->params->beta;
  std::vector<int> doc_dist;
  std::vector<Token> tmp_token(n_topics);

  // deal with long docs
  #pragma omp parallel for schedule(dynamic) \
    firstprivate(doc_dist, tmp_token)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    if (!is_long_doc[doc])
      continue;
    int thread = OMP_GET_THREAD_ID();
    int N = corpus.doc_offset[doc + 1] - corpus.doc_offset[doc];
    int offset = corpus.doc_offset[doc];
    doc_dist.clear();
    doc_dist.resize(n_topics);
    tmp_token.clear();

    for (int i = 0; i < N; ++i) {
      tmp_token.push_back(topics[corpus.doc_to_word[offset + i]]);
      doc_dist[tmp_token[i].topic]++;
    }

    // re-assign topic for each token in this LONG doc.
    for (int i = 0; i < N; ++i) {
      Token &tok = tmp_token[i];
      int topic = tok.topic;
      doc_dist[topic]--;
      if (update)
        topic_dist[topic]--;

      for (int m = 0; m < MH_STEP; ++m) {
        int new_topic = tok.mh_step[m];
        float Cdj = doc_dist[new_topic] + alpha; // q_dis
        float Cdi = doc_dist[topic] + alpha;
        float Cj = topic_dist[new_topic] + corpus.n_words * beta;
        float Ci = topic_dist[topic] + corpus.n_words * beta;
        float prob = Cdj * Ci * rand[thread].MAX_N;
        topic = rand[thread].RandInt() * Cdi * Cj < prob ? new_topic : topic;
      }
      if (update)
        topic_dist[topic]++;
      doc_dist[topic]++;
      tok.topic = topic;
    }

    unsigned prob = (n_topics * alpha) / (n_topics * alpha + N) * rand[thread].MAX_N;
    for (int i = 0; i < N; ++i) {
      Token &tok = tmp_token[i];
      for (int m = 0; m < MH_STEP; ++m) {
        int a = rand[thread].RandInt(n_topics);
        int b = tmp_token[rand[thread].RandInt(N)].topic;
        tok.mh_step[m] = rand[thread].RandInt() < prob ? a : b;
      }
    }
    for (int i = 0; i < N; ++i) {
      topics[corpus.doc_to_word[offset + i]] = tmp_token[i];
    }
  }
}

double LDA::Loglikelihood(const Corpus &corpus) {
  int n_topics = this->params->num_topics;
  float alpha = this->params->alpha;
  float beta = this->params->beta;
  double llh = 0;
  std::vector<int> doc_dist;

  llh += corpus.n_docs * (lgamma(n_topics * alpha) - n_topics * lgamma(alpha));
  #pragma omp parallel for schedule(dynamic) private(doc_dist) reduction(+:llh)
  for (int doc = 0; doc < corpus.n_docs; ++doc) {
    llh -= lgamma(corpus.GetDocSize(doc) + n_topics * alpha);
    if (is_long_doc[doc]) {
      doc_dist.clear();
      doc_dist.resize(n_topics);
      for (int i = corpus.doc_offset[doc]; i < corpus.doc_offset[doc + 1]; ++i) {
        doc_dist[topics[corpus.doc_to_word[i]].topic]++;
      }
      for (int topic = 0; topic < n_topics; ++topic) {
        llh += lgamma(doc_dist[topic] + alpha);
      }
    } else {
      for (auto &item : doc_topic_dist[doc].GetItem()) {
        llh += lgamma(item.count + alpha);
      }
      llh += lgamma(alpha) * (n_topics - doc_topic_dist[doc].GetItem().size());
    }
  }

  llh += n_topics * (lgamma(beta * corpus.n_words) - corpus.n_words * lgamma(beta));
  #pragma omp parallel for schedule(dynamic) reduction(+:llh)
  for (int word = 0; word < corpus.n_words; ++word) {
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += lgamma(word_topic_dist[word * n_topics + topic] + beta);
    }
  }
  for (int topic = 0; topic < n_topics; ++topic) {
    llh -= lgamma(topic_dist[topic] + corpus.n_words * beta);
  }
  return llh;
}

} // namespace lda
} // namespace ml
} // namespace hetu
