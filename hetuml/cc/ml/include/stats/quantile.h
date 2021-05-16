#ifndef __HETU_ML_STATS_QUANTILE_H_
#define __HETU_ML_STATS_QUANTILE_H_

#include "common/logging.h"
#include "common/math.h"
#include <chrono>
#include <random>
#include <algorithm>

namespace hetu { 
namespace ml {

template <typename T>
class QuantileSummary {
public:

  virtual ~QuantileSummary() {}

  virtual T Query(double fraction) = 0;

  virtual void Query(const std::vector<double>& fractions, 
      std::vector<T>& ret) = 0;

  virtual void Query(int even_partition, std::vector<T>& ret) = 0;
  
  virtual bool TryDistinct(int max_num_items, std::vector<T>& ret) = 0;
};

template <typename T>
class QuantileSketch {
public:

  QuantileSketch() {
    this->n = 0;
    this->min_value = std::numeric_limits<T>::max();
    this->max_value = std::numeric_limits<T>::min();
  }

  virtual ~QuantileSketch() {}

  virtual QuantileSummary<T>* MakeSummary(double resolution = 0.01) = 0;

  virtual void Update(const T& value) = 0;

  inline bool empty() const { return n == 0; }

  inline int64_t get_n() const { return n; }

  inline const T& get_min_value() const { return min_value; }

  inline const T& get_max_value() const { return max_value; }

protected:
  int64_t n;
  T min_value;
  T max_value;
};


/**
* \brief Histogram-based quantile sketch.
*        Acharya et al. Fast and Near–Optimal Algorithms for Approximating Distributions by Histograms
*/
template <typename T>
class HistQuantileSummary : public QuantileSummary<T> {
public:

  HistQuantileSummary(std::vector<T>& splits): splits{ std::move(splits) } {}

  T Query(double fraction) {
    if (fraction == 0) return splits[0];
    if (fraction == 1) return splits[splits.size() - 1];
    int64_t pos = round(fraction * splits.size());
    return splits[MIN(pos, splits.size() - 1)];
  }

  void Query(const std::vector<double>& fractions, 
      std::vector<T>& ret) {
    ret.resize(fractions.size());
    for (uint32_t i = 0; i < fractions.size(); i++) {
      ret[i] = Query(fractions[i]);
    }
  }

  void Query(int even_partition, std::vector<T>& ret) {
    ret.resize(even_partition);
    ret[0] = splits[0];
    double cur_frac = 0, step = 1.0 / even_partition;
    for (int i = 1; i < even_partition; i++) {
      cur_frac += step;
      ret[i] = Query(cur_frac);
    }
  }
  
  bool TryDistinct(int max_num_items, std::vector<T>& ret) {
    int cnt = 1;
    for (int i = 1; i < splits.size(); i++) {
      ASSERT(splits[i] >= splits[i - 1]) << "Error summary: " 
        << splits[i] << " is smaller than " << splits[i - 1];
      if (splits[i] != splits[i - 1]) {
        if ((++cnt) > max_num_items)
          return false;
      }
    }

    ret.resize(cnt);
    if (cnt != splits.size()) {
      ret[0] = splits[0];
      int index = 1;
      for (int i = 1; i < splits.size(); i++) {
        if (splits[i] != splits[i - 1])
          ret[index++] = splits[i];
      }
    } else {
      ret.insert(ret.begin(), splits.begin(), splits.end());
    }
    return true;
  }

private:
  const std::vector<T> splits;
};

template <typename T>
class HistQuantileSketch : public QuantileSketch<T> {
public:
  
  HistQuantileSketch(double sp_ratio, int64_t max_buffer_size)
  : sp_ratio(sp_ratio), max_buffer_size(max_buffer_size),
    distribution(0.0, 1.0) {
    // engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    engine.seed(1234L);
  }

  HistQuantileSketch(int64_t max_buffer_size)
  : HistQuantileSketch(1.0, max_buffer_size) {}

  HistQuantileSketch()
  : HistQuantileSketch(-1) {}

  void Update(const T& value) {
    this->n++;
    this->min_value = MIN(this->min_value, value);
    this->max_value = MAX(this->max_value, value);
    if (sp_ratio == 1 || distribution(engine) < sp_ratio) {
      if (max_buffer_size <= 0 || buffer.size() < max_buffer_size) {
        buffer.push_back(value);
      } else {
        int64_t pos = round(distribution(engine) * max_buffer_size);
        buffer[MIN(pos, max_buffer_size - 1)] = value;
      }
    }
  }

  QuantileSummary<T>* MakeSummary(double resolution = 0.01) {
    ASSERT(!this->empty()) << "Cannot make summary for an empty sketch";
    if (buffer.empty()) {
      std::vector<T> splits = { this->min_value, this->max_value };
      return new HistQuantileSummary<T>(splits);
    }
    // pre-compute the errors
    uint32_t size = buffer.size();
    std::sort(buffer.begin(), buffer.end());
    std::vector<T> r(buffer.size()), t(buffer.size());
    r[0] = buffer[0];
    t[0] = buffer[0] * buffer[0];
    for (uint32_t i = 1; i < size; i++) {
      r[i] = r[i - 1] + buffer[i];
      t[i] = t[i - 1] + buffer[i] * buffer[i];
    }
    // find split points
    uint32_t num_splits = size, num_bins = ceil(1 / resolution);
    std::vector<uint32_t> split_indexes(size);
    std::iota(split_indexes.begin(), split_indexes.end(), 0);
    std::vector<T> errors(size);
    while (num_splits > num_bins) {
      for (uint32_t i = 0; i < (num_splits >> 1); i++) {
        uint32_t L = split_indexes[2 * i];
        uint32_t R = (2 * i + 2 >= num_splits) ? size 
                    : split_indexes[2 * i + 2] - 1;
        T sum_l1 = r[R] - (L == 0 ? 0 : r[L - 1]);
        T sum_l2 = t[R] - (L == 0 ? 0 : t[L - 1]);
        T mean = sum_l1 / (R - L + 1);
        errors[i] = sum_l2 + mean * mean * (R - L + 1) - 2 * mean * sum_l1;
      }

      uint32_t num_thr = (num_bins >> 1) - (IS_ODD(num_splits) ? 1 : 0);
      T threshold = SelectKthLargest<T>(errors, num_thr);

      uint32_t new_num_splits = 0;
      for (uint32_t i = 0; i < (num_splits >> 1); i++) {
        if (errors[i] < threshold) {
          split_indexes[new_num_splits++] = split_indexes[2 * i];
        } else {
          split_indexes[new_num_splits++] = split_indexes[2 * i];
          split_indexes[new_num_splits++] = split_indexes[2 * i + 1];
        }
      }

      if (IS_ODD(num_splits)) 
        split_indexes[new_num_splits++] = split_indexes[num_splits - 1];
      if (num_splits == new_num_splits) 
        break; // true if there are many frequently-occurred items
      else
        num_splits = new_num_splits;
    }
    // make summary
    num_bins = num_splits;
    std::vector<T> splits(num_bins + 1);
    splits[0] = this->min_value;
    for (uint32_t i = 1; i < num_bins; i++)
      splits[i] = buffer[split_indexes[i]];
    splits[num_bins] = this->max_value;
    return new HistQuantileSummary<T>(splits);
  }

private:
  std::vector<T> buffer;
  double sp_ratio;
  int64_t max_buffer_size;
  std::default_random_engine engine;
  std::uniform_real_distribution<double> distribution;
};

} // namespace ml
} // namespace hetu

#endif // __HETU_ML_STATS_QUANTILE_H_
