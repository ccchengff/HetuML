#ifndef __HETU_ML_MODEL_LDA_COMMON_FTREE_H_
#define __HETU_ML_MODEL_LDA_COMMON_FTREE_H_

#include <vector>

namespace hetu { 
namespace ml {
namespace lda {

/* A Scalable Asynchronous Distributed Algorithm for 
Topic Modeling. Hsiang-Fu Yu et al. WWW2015 */
class FTree {
private:
  std::vector<float> value;
  int n = 0;
  int topic_num = 0;

public:
  float Get(int i) { return value[n + i]; }

  float Sum() { return value[1]; }

  void Set(int i, float val) { value[n + i] = val; }

  // build must be called after values set
  void Build() {
    for (int i = n - 1; i >= 1; --i) {
      value[i] = value[i + i] + value[i + i + 1];
    }
  }

  void Update(int i, float val) {
    i += n;
    value[i] = val;
    while (i > 1) {
      i >>= 1;
      value[i] = value[i + i] + value[i + i + 1];
    }
  }

  int Sample(float prob) {
    int i = 1;
    while (i < n) {
      if (prob < value[i + i]) {
        i = i + i;
      } else {
        prob -= value[i + i];
        i = i + i + 1;
      }
    }
    return std::min(i - n, topic_num - 1);
  }

  void Resize(int size) {
    n = 1;
    while (n < size)
      n <<= 1;
    value.resize(n * 2);
  }

  explicit FTree(int size) : topic_num(size) { Resize(size); }

  FTree() = default;
};

} // namespace lda
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LDA_COMMON_FTREE_H_
