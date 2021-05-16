#ifndef __HETU_ML_MODEL_LDA_COMMON_SPARSE_COUNTER_H_
#define __HETU_ML_MODEL_LDA_COMMON_SPARSE_COUNTER_H_

#include <mutex>
#include <vector>

namespace hetu { 
namespace ml {
namespace lda {

/*
 * used in Sparse-A sampler, fast for enumerating. 
 * Here used in F+LDA.
 */
struct CountItem {
  int item;
  int count;

  CountItem(int it, int cnt) : item(it), count(cnt) {}
  CountItem() = default;
};

class SparseCounter {
public:
  void Dec(int idx) {
    int i = index[idx];
    item[i].count--;
    if (item[i].count == 0) {
      item[i] = *item.rbegin();
      index[item[i].item] = i;
      item.pop_back();
    }
  }

  void Inc(int idx) {
    int i = index[idx];
    if (item.size() <= i || item[i].item != idx) {
      SetNew(idx, 1);
    } else {
      item[i].count++;
    }
  }

  void SetNew(int idx, int cnt) {
    index[idx] = static_cast<int>(item.size());
    item.emplace_back(CountItem(idx, cnt));
  }

  int Count(int idx) {
    if (item.size() <= index[idx] || item[index[idx]].item != idx) {
      return 0;
    }
    return item[index[idx]].count;
  }

  void Resize(int size) {
    item.reserve(size);
    index.resize(size);
  }

  void Clear() {
    int size = index.size();
    item.clear();
    index.clear();
    item.reserve(size);
    index.resize(size);
  }

  void Swap(SparseCounter &another) {
    item.swap(another.item);
    index.swap(another.index);
  }

  const std::vector<CountItem> &GetItem() { return item; }

  void Lock() { mtx.lock(); }
  void Unlock() { mtx.unlock(); }

  SparseCounter(SparseCounter &&a) {
    item.swap(a.item);
    index.swap(a.index);
  }
  SparseCounter(int size) { Resize(size); }
  SparseCounter() = default;

private:
  std::vector<CountItem> item;
  std::vector<int> index;
  std::mutex mtx;
};

} // namespace lda
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LDA_COMMON_SPARSE_COUNTER_H_
