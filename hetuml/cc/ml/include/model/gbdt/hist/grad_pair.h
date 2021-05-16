#ifndef __HETU_ML_MODEL_GBDT_HIST_GRAD_PAIR_H_
#define __HETU_ML_MODEL_GBDT_HIST_GRAD_PAIR_H_

#include "common/logging.h"
#include "model/common/argparse.h"
#include <iterator>
#include <vector>
#include <algorithm>

namespace hetu { 
namespace ml {
namespace gbdt {

class GradPair {
public:

  virtual ~GradPair() {}
  
  virtual void PlusBy(const GradPair& other) = 0;

  virtual void SubtractBy(const GradPair& other) = 0;

  virtual void TimesBy(double x) = 0;

  virtual void set(const GradPair& other) = 0;

  virtual void get(std::vector<double>& values) const = 0;

  virtual void set(const std::vector<double>& values) = 0;

  virtual void get(double* values) const = 0;

  virtual void set(const double* values) = 0;

  virtual GradPair* copy() const = 0;

  virtual void clear() = 0;

  virtual GradPair* zeros_like() const = 0;

  virtual bool is_multi() const = 0;

  virtual void Print(std::ostream& os) const = 0;

  virtual std::string ToString() const = 0;

  virtual void FromString(const std::string& str) = 0;

  friend std::ostream& operator<<(std::ostream& os, const GradPair& gp) {
    gp.Print(os);
    return os;
  }
};

class BinaryGradPair : public GradPair {
public:

  BinaryGradPair(): m1(0.0), m2(0.0) {}

  BinaryGradPair(double m1, double m2): m1(m1), m2(m2) {}

  ~BinaryGradPair() {}

  inline void PlusBy(const GradPair& other) {
    this->m1 += ((const BinaryGradPair&) other).m1;
    this->m2 += ((const BinaryGradPair&) other).m2;
  }

  inline void PlusBy(double m1, double m2) {
    this->m1 += m1;
    this->m2 += m2;
  }

  inline void SubtractBy(const GradPair& other) {
    this->m1 -= ((const BinaryGradPair&) other).m1;
    this->m2 -= ((const BinaryGradPair&) other).m2;
  }

  inline void SubtractBy(double m1, double m2) {
    this->m1 -= m1;
    this->m2 -= m2;
  }

  inline void TimesBy(double x) {
    this->m1 *= x;
    this->m2 *= x;
  }

  inline void set(const GradPair& other) {
    const BinaryGradPair& binary = (const BinaryGradPair&) other;
    set(binary.m1, binary.m2);
  }

  inline void set(double m1, double m2) { 
    this->m1 = m1;
    this->m2 = m2;
  }

  inline void get(std::vector<double>& values) const {
    values.resize(2);
    values[0] = this->m1;
    values[1] = this->m2;
  }

  inline void get(double* values) const {
    values[0] = this->m1;
    values[1] = this->m2;
  }

  inline void set(const std::vector<double>& values) {
    this->m1 = values[0];
    this->m2 = values[1];
  }

  inline void set(const double* values) {
    this->m1 = values[0];
    this->m2 = values[1];
  }

  inline GradPair* copy() const {
    return new BinaryGradPair(m1, m2);
  }

  inline void clear() {
    this->m1 = 0.0;
    this->m2 = 0.0;
  }

  inline GradPair* zeros_like() const {
    return new BinaryGradPair();
  }

  inline bool is_multi() const {
    return false;
  }

  inline const double get_m1() const { return m1; }

  inline const double get_m2() const { return m2; }

  void Print(std::ostream& os) const {
    os << "{ m1[" << this->m1 << "], m2[" << this->m2 << "] }";
  }

  std::string ToString() const {
    std::ostringstream os;
    Print(os);
    return os.str();
  }

  void FromString(const std::string& str) {
    size_t start = 0, end = 0;
    start = argparse::GetOffset(str, "m1[", end + 1);
    end = argparse::GetOffset(str, "]", start + 1) - 1;
    this->m1 = argparse::Parse<double>(str.substr(start, end - start));
    start = argparse::GetOffset(str, "m2[", end + 1);
    end = argparse::GetOffset(str, "]", start + 1) - 1;
    this->m2 = argparse::Parse<double>(str.substr(start, end - start));
  }

private:
  double m1;
  double m2;
};

class MultiGradPair : public GradPair {
public:

  MultiGradPair(int num_classes)
  : MultiGradPair(num_classes, num_classes) {}

  MultiGradPair(uint32_t m1_size, uint32_t m2_size)
  : m1(m1_size, 0), m2(m2_size, 0) {}

  MultiGradPair(const std::vector<double>& m1, const std::vector<double>& m2)
  : m1(m1), m2(m2) {}

  ~MultiGradPair() {}

  inline void PlusBy(const GradPair& other) {
    const std::vector<double>& m1 = ((const MultiGradPair&) other).m1;
    for (int i = 0; i < this->m1.size(); i++)
      this->m1[i] += m1[i];
    const std::vector<double>& m2 = ((const MultiGradPair&) other).m2;
    for (int i = 0; i < this->m2.size(); i++)
      this->m2[i] += m2[i];
  }

  inline void PlusBy(const std::vector<double>& m1, 
                     const std::vector<double>& m2, 
                     int offset) {
    for (int i = 0; i < this->m1.size(); i++) {
      this->m1[i] += m1[offset + i];
      this->m2[i] += m2[offset + i];
    }
  }

  inline void PlusBy(const std::vector<double>& m1, int m1_offset, 
                     const std::vector<double>& m2, int m2_offset) {
    for (int i = 0; i < this->m1.size(); i++) 
      this->m1[i] += m1[m1_offset + i];
    for (int i = 0; i < this->m2.size(); i++) 
      this->m2[i] += m2[m2_offset + i];
  }

  inline void SubtractBy(const GradPair& other) {
    const std::vector<double>& m1 = ((const MultiGradPair&) other).m1;
    for (int i = 0; i < this->m1.size(); i++)
      this->m1[i] -= m1[i];
    const std::vector<double>& m2 = ((const MultiGradPair&) other).m2;
    for (int i = 0; i < this->m2.size(); i++)
      this->m2[i] -= m2[i];
  }

  inline void SubtractBy(const std::vector<double>& m1, 
                         const std::vector<double>& m2, 
                         int offset) {
    for (int i = 0; i < this->m1.size(); i++) {
      this->m1[i] -= m1[offset + i];
      this->m2[i] -= m2[offset + i];
    }
  }

  inline void SubtractBy(const std::vector<double>& m1, int m1_offset, 
                         const std::vector<double>& m2, int m2_offset) {
    for (int i = 0; i < this->m1.size(); i++) 
      this->m1[i] -= m1[m1_offset + i];
    for (int i = 0; i < this->m2.size(); i++) 
      this->m2[i] -= m2[m2_offset + i];
  }

  inline void TimesBy(double x) {
    for (int i = 0; i < this->m1.size(); i++)
      this->m1[i] *= x;
    for (int i = 0; i < this->m2.size(); i++)
      this->m2[i] *= x;
  }

  inline void set(const GradPair& other) {
    const MultiGradPair& multi = (const MultiGradPair&) other;
    std::copy(multi.m1.begin(), multi.m1.end(), this->m1.begin());
    std::copy(multi.m2.begin(), multi.m2.end(), this->m2.begin());
  }

  inline void set(const std::vector<double>& m1, 
                  const std::vector<double>& m2, 
                  int offset) {
    set(m1, offset, m2, offset);
  }

  inline void set(const std::vector<double>& m1, int m1_offset, 
                  const std::vector<double>& m2, int m2_offset) {
    std::copy(
      m1.begin() + m1_offset, 
      m1.begin() + m1_offset + this->m1.size(), 
      this->m1.begin()
    );
    std::copy(
      m2.begin() + m2_offset, 
      m2.begin() + m2_offset + this->m2.size(), 
      this->m2.begin()
    );
  }

  inline GradPair* copy() const {
    return new MultiGradPair(m1, m2);
  }

  inline void clear() {
    std::fill(this->m1.begin(), this->m1.end(), 0);
    std::fill(this->m2.begin(), this->m2.end(), 0);
  }

  inline GradPair* zeros_like() const {
    return new MultiGradPair(m1.size(), m2.size());
  }

  inline bool is_multi() const {
    return true;
  }

  inline void get(std::vector<double>& values) const {
    values.resize(this->m1.size() + this->m2.size());
    std::copy(this->m1.begin(), this->m1.end(), values.begin());
    std::copy(this->m2.begin(), this->m2.end(), 
      values.begin() + this->m1.size());
  }

  inline void get(double* values) const {
    std::copy(this->m1.begin(), this->m1.end(), values);
    std::copy(this->m2.begin(), this->m2.end(), values + this->m1.size());

  }

  inline void set(const std::vector<double>& values) {
    std::copy(
      values.begin(), 
      values.begin() + this->m1.size(), 
      this->m1.begin()
    );
    std::copy(
      values.begin() + this->m1.size(), 
      values.begin() + this->m1.size() + this->m2.size(), 
      this->m2.begin()
    );
  }

  inline void set(const double* values) {
    std::copy(
      values, 
      values + this->m1.size(), 
      this->m1.begin()
    );
    std::copy(
      values + this->m1.size(), 
      values + this->m1.size() + this->m2.size(), 
      this->m2.begin()
    );
  }

  inline const std::vector<double>& get_m1() const { return m1; }

  inline const std::vector<double>& get_m2() const { return m2; }

  void Print(std::ostream& os) const {
    os << "{ m1" << this->m1 << ", m2" << this->m2 << " }";
  }

  std::string ToString() const {
    std::ostringstream os;
    Print(os);
    return os.str();
  }

  void FromString(const std::string& str) {
    size_t start = 0, end = 0;
    start = argparse::GetOffset(str, "m1", end + 1);
    end = argparse::GetOffset(str, "]", start + 1);
    argparse::ParseVector<double>(str.substr(start, end - start), this->m1);
    start = argparse::GetOffset(str, "m2", end + 1);
    end = argparse::GetOffset(str, "]", start + 1);
    argparse::ParseVector<double>(str.substr(start, end - start), this->m2);
  }

private:
  std::vector<double> m1;
  std::vector<double> m2;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_GBDT_HIST_GRAD_PAIR_H_
