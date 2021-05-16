#ifndef __HETU_ML_MODEL_LDA_COMMON_CORPUS_H_
#define __HETU_ML_MODEL_LDA_COMMON_CORPUS_H_

#include "common/logging.h"
#include "model/lda/common/random.h"
#include <cmath>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hetu { 
namespace ml {
namespace lda {

class Corpus {
public:
  Corpus(int n_words): n_words(n_words), n_docs(0), n_tokens(0) {}
  
  ~Corpus() {}
  
  // file format: docId wordId wordId wordId...
  void ReadFromFile(const std::string& doc_path, size_t rank = 0, 
                    size_t num_workers = 1) {
    TIK(load);
    std::ifstream fin(doc_path, std::ifstream::in);
    ASSERT(fin) << "Failed to open " << doc_path;
    
    std::string doc;
    int read_buffer_size = 1 << 20;
    char read_buffer[read_buffer_size];
    std::vector<int> words(256);
    size_t line_num = 0;
    // read documents
    while (fin >> doc) {
      fin.getline(read_buffer, read_buffer_size, '\n');
      line_num++;
      if (num_workers > 1 && (line_num - 1) % num_workers != rank)
        continue;
      // read words
      std::stringstream doc_s(read_buffer);
      words.clear();
      std::string word;
      std::string did;
      doc_s >> did; // dummy doc Id
      while (doc_s >> word) {
        words.push_back(std::stoi(word));
      }
      n_docs++;
      doc_offset.push_back(docs.size());
      docs.insert(docs.end(), words.begin(), words.end());
    }
    fin.close();
    FinishAdding();

    TOK(load);
    HML_LOG_INFO << "Read doc from " << doc_path 
      << " cost " << COST_MSEC(load) << " ms, "
      << "#docs[" << n_docs << "] " 
      << "#words[" << n_words << "] "
      << "#tokens[" << n_tokens << "]";
  }

  void FinishAdding() {
    n_tokens = docs.size();
    words.reserve(n_tokens);
    doc_offset.push_back(docs.size());
    word_offset.resize(n_words + 1);
    doc_to_word.reserve(n_tokens);
    word_to_doc.reserve(n_tokens);

    for (int i = 0; i < n_tokens; ++i)
      word_offset[docs[i]]++;
    for (int i = 1; i < n_words; ++i)
      word_offset[i] += word_offset[i - 1];
    word_offset[n_words] = word_offset[n_words - 1];
    for (int d = 0; d < n_docs; ++d) {
      for (int i = doc_offset[d]; i < doc_offset[d + 1]; ++i) {
        int w = docs[i];
        words[--word_offset[w]] = d;
        doc_to_word[i] = word_offset[w];
        word_to_doc[word_offset[w]] = i;
      }
    }
    // shuffle words_
    for (int word = 0; word < n_words; ++word) {
      int begin = word_offset[word];
      int end = word_offset[word + 1];
      int N = end - begin;
      Random rand;
      for (int i = 0; i < 2 * N; ++i) {
        int a = rand.RandInt(N) + begin;
        int b = rand.RandInt(N) + begin;
        std::swap(words[a], words[b]);
        std::swap(word_to_doc[a], word_to_doc[b]);
        doc_to_word[word_to_doc[a]] = a;
        doc_to_word[word_to_doc[b]] = b;
      }
    }
  }
  
  inline int GetWordSize(int id) const {
    return word_offset[id + 1] - word_offset[id];
  }
  
  inline int GetDocSize(int id) const { 
    return doc_offset[id + 1] - doc_offset[id]; 
  }

  // Number of documents
  int n_docs;
  // Number of vocabulary
  int n_words;
  // Number of tokens
  int n_tokens;
  // Document Ids organized by word
  std::vector<int> words; // #tokens
  // Word Ids organized by document
  std::vector<int> docs; // #tokens
  // index map from docs to words
  std::vector<int> doc_to_word; //#tokens
  // index map from words to docs
  std::vector<int> word_to_doc; //#tokens
  // Cumulative counts for the number of words
  std::vector<int> word_offset; //# words
  // Cumulative counts for the number of documents
  std::vector<int> doc_offset; // #docs
};

} // namespace lda
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LDA_COMMON_CORPUS_H_
