#ifndef __HETU_ML_MODEL_GBDT_HIST_HISTOGRAM_H_
#define __HETU_ML_MODEL_GBDT_HIST_HISTOGRAM_H_

#include "model/gbdt/hist/grad_pair.h"

namespace hetu { 
namespace ml {
namespace gbdt {

class Histogram;
typedef std::vector<std::shared_ptr<Histogram>> NodeHist;

class Histogram {
public:
  Histogram(int num_bins, int bin_size)
  : Histogram(num_bins, bin_size, bin_size) {}

  Histogram(int num_bins, int m1_size, int m2_size)
  : num_bins(num_bins), m1_size(m1_size), m2_size(m2_size) {
    if (m1_size == 1) {
      m1.resize(num_bins, 0.0);
      m2.resize(num_bins, 0.0);
    } else {
      ASSERT(m1_size <= m2_size) << "Size of m1[" << m1_size 
        << "] is larger than m2[" << m2_size << "]";
      m1.resize(num_bins * m1_size, 0.0);
      m2.resize(num_bins * m2_size, 0.0);
    }
  }
  
  inline void Accumulate(int index, double m1, double m2) {
    this->m1[index] += m1;
    this->m2[index] += m2;
  }

  inline void Accumulate(int index, const std::vector<double>& m1, 
                         const std::vector<double>& m2) {
    Accumulate(index, m1, m2, 0);
  }

  inline void Accumulate(int index, const std::vector<double>& m1, 
                         const std::vector<double>& m2, int offset) {
    int bin_offset = index * m1_size;
    for (int i = 0; i < m1_size; i++) {
      this->m1[bin_offset + i] += m1[offset + i];
      this->m2[bin_offset + i] += m2[offset + i];
    }
  }

  inline void Accumulate(int index, 
                         const std::vector<double>& m1, int m1_offset, 
                         const std::vector<double>& m2, int m2_offset) {
    int m1_bin_offset = index * m1_size;
    int m2_bin_offset = index * m2_size;
    for (int i = 0; i < m1_size; i++) 
      this->m1[m1_bin_offset + i] += m1[m1_offset + i];
    for (int i = 0; i < m2_size; i++) 
      this->m2[m2_bin_offset + i] += m2[m2_offset + i];
  }

  inline void Accumulate(int index, GradPair& gp) {
    if (m1_size == 1) {
      BinaryGradPair& binary = (BinaryGradPair&) gp;
      Accumulate(index, binary.get_m1(), binary.get_m2());
    } else {
      MultiGradPair& multi = (MultiGradPair&) gp;
      Accumulate(index, multi.get_m1(), multi.get_m2());
    }
  }

  inline void FillRemain(const GradPair& sum_gp, int index) {
    if (m1_size == 1) {
      // binary-class or multi-class-multi-tree
      double my_m1 = 0, my_m2 = 0;
      for (double v1 : m1) my_m1 += v1;
      for (double v2 : m2) my_m2 += v2;
      const BinaryGradPair& binary = (const BinaryGradPair&) sum_gp;
      m1[index] += binary.get_m1() - my_m1;
      m2[index] += binary.get_m2() - my_m2;
    } else if (m1_size == m2_size) {
      // multi-class assuming m2 is diagonal
      std::vector<double> my_m1(m1_size, 0);
      std::vector<double> my_m2(m1_size, 0);
      for (int i = 0; i < num_bins; i++) {
        for (int k = 0; k < m1_size; k++) {
          my_m1[k] += m1[i * m1_size + k];
          my_m2[k] += m2[i * m1_size + k];
        }
      }
      const MultiGradPair& multi = (const MultiGradPair&) sum_gp;
      const auto& sum_m1 = multi.get_m1();
      const auto& sum_m2 = multi.get_m2();
      int bin_offset = index * m1_size;
      for (int k = 0; k < m1_size; k++) {
        m1[bin_offset + k] += sum_m1[k] - my_m1[k];
        m2[bin_offset + k] += sum_m2[k] - my_m2[k];
      }
    } else {
      // multi-class when m1 and m2 have different sizes
      std::vector<double> my_m1(m1_size, 0);
      std::vector<double> my_m2(m2_size, 0);
      for (int i = 0; i < num_bins; i++) {
        for (int k = 0; k < m1_size; k++) 
          my_m1[k] += m1[i * m1_size + k];
        for (int k = 0; k < m2_size; k++) 
          my_m2[k] += m2[i * m2_size + k];
      }
      const MultiGradPair& multi = (const MultiGradPair&) sum_gp;
      const auto& sum_m1 = multi.get_m1();
      const auto& sum_m2 = multi.get_m2();
      int m1_bin_offset = index * m1_size;
      int m2_bin_offset = index * m1_size;
      for (int k = 0; k < m1_size; k++) 
        m1[m1_bin_offset + k] += sum_m1[k] - my_m1[k];
      for (int k = 0; k < m2_size; k++) 
        m2[m2_bin_offset + k] += sum_m2[k] - my_m2[k];
    }
  }

  inline void PlusBy(Histogram& other) {
    for (int i = 0; i < this->m1.size(); i++) 
      this->m1[i] += other.m1[i];
    for (int i = 0; i < this->m2.size(); i++) 
      this->m2[i] += other.m2[i];
  }

  inline void SubtractBy(Histogram& other) {
    for (int i = 0; i < this->m1.size(); i++) 
      this->m1[i] -= other.m1[i];
    for (int i = 0; i < this->m2.size(); i++) 
      this->m2[i] -= other.m2[i];
  }

  inline void GetBin(int index, GradPair& gp) const {
    if (m1_size == 1) {
      ((BinaryGradPair&) gp).set(m1[index], m2[index]);
    } else if (m1_size == m2_size) {
      ((MultiGradPair&) gp).set(m1, m2, index * m1_size);
    } else {
      ((MultiGradPair&) gp).set(m1, index * m1_size, m2, index * m2_size);
    }
  }

  void Scan(int index, GradPair& left, GradPair& right) const {
    if (m1_size == 1) {
      ((BinaryGradPair&) left).PlusBy(m1[index], m2[index]);
      ((BinaryGradPair&) right).SubtractBy(m1[index], m2[index]);
    } else if (m1_size == m2_size) {
      ((MultiGradPair&) left).PlusBy(m1, m2, index * m1_size);
      ((MultiGradPair&) right).SubtractBy(m1, m2, index * m1_size);
    } else {
      ((MultiGradPair&) left).PlusBy(
        m1, index * m1_size, m2, index * m2_size);
      ((MultiGradPair&) right).SubtractBy(
        m1, index * m1_size, m2, index * m2_size);
    }
  }

  inline GradPair* Sum() const { return Sum(0, num_bins); }

  inline GradPair* Sum(int start, int end) const {
    if (m1_size == 1) {
      double sum_m1 = 0, sum_m2 = 0;
      for (int bin_id = start; bin_id < end; bin_id++) {
        sum_m1 += m1[bin_id];
        sum_m2 += m2[bin_id];
      }
      return new BinaryGradPair(sum_m1, sum_m2);
    } else if (m1_size == m2_size) {
      MultiGradPair *multi = new MultiGradPair(m1_size);
      for (int bin_id = start; bin_id < end; bin_id++) {
        multi->PlusBy(m1, m2, bin_id * m1_size);
      }
      return multi;
    } else {
      MultiGradPair *multi = new MultiGradPair(m1_size, m2_size);
      for (int bin_id = start; bin_id < end; bin_id++) {
        multi->PlusBy(m1, bin_id * m1_size, m2, bin_id * m2_size);
      }
      return multi;
    }
  }

  inline void clear() {
    std::fill(this->m1.begin(), this->m1.end(), 0);
    std::fill(this->m2.begin(), this->m2.end(), 0);
  }

  inline int get_num_bins() const { return num_bins; }

  inline int get_m1_size() const { return m1_size; }

  inline int get_m2_size() const { return m2_size; }

  inline std::vector<double>& get_m1() { return m1; }

  inline const std::vector<double>& get_m1() const { return m1; }

  inline std::vector<double>& get_m2() { return m2; }

  inline const std::vector<double>& get_m2() const { return m2; }

private:
  int num_bins;
  int m1_size;
  int m2_size;
  std::vector<double> m1;
  std::vector<double> m2;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_HIST_HISTOGRAM_H_
