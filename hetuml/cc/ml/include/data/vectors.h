  
#ifndef __HETU_ML_DATA_VECTORS_H_
#define __HETU_ML_DATA_VECTORS_H_

#include "common/logging.h"
// #include <cmath>

/**
* \brief AVector serve as a wrapper for std::vector<Val> to support sparse/dense operations.
*        Programmers take care of the memory of vectors.
*        Attention: No huge memory allocation in this class.
*/
template <typename Val> class DenseVector;
template <typename Val> class SparseVector;

template <typename Val> class AVector {
public:
  int dim;
  AVector(int dim): dim(dim) {}
  ~AVector() {}
  virtual Val dot(const SparseVector<Val>& other) const = 0;
  virtual Val dot(const DenseVector<Val>& other) const = 0;
  virtual void axp0(const SparseVector<Val>& other, Val a) = 0;
  virtual void axp0(const DenseVector<Val>& other, Val a) = 0;
  virtual void clear() = 0;
  virtual AVector<Val>* copy() const = 0;
  virtual bool is_dense() const = 0;
};

template <typename Val> class DenseVector : public AVector<Val> {
public:
  Val *values;
  
  DenseVector(Val* vals, int dim): AVector<Val>(dim), values(vals) {}

  DenseVector(int dim, Val init_value = 0)
  : DenseVector<Val>(new Val[dim], dim) {
    std::fill(this->values, this->values + this->dim, init_value);
  }

  DenseVector(const DenseVector<Val>& other)
  : DenseVector<Val>(new Val[other.dim], other.dim) {
    std::copy(other.values, other.values + other.dim, this->values);
  }
  
  ~DenseVector() { if (values != nullptr) delete[]values; }

  Val dot(const SparseVector<Val>& other) const override { 
    return other.dot(*this); 
  }

  // dense dot dense
  Val dot(const DenseVector<Val>& other) const override {
    // assert(this->dim == other.dim);
    Val res = 0;
    auto i_values = this->values;
    const auto o_values = other.values;
    for (int i = 0; i < this->dim; i++) {
      res += i_values[i] * o_values[i];
    }
    return res;
  }

  // this += a * other;
  void axp0(const SparseVector<Val>& other, Val a) override {
    // assert(this->dim == other.dim);
    auto i_values = this->values;
    const auto o_indices = other.indices;
    const auto o_values = other.values;
    int nnz = other.nnz;
    for (int i = 0; i < nnz; i++) {
      i_values[o_indices[i]] += a * o_values[i];
    }
  }

  // this += a * other
  void axp0(const DenseVector<Val>& other, Val a) override {
    // assert(this->dim == other.dim);
    auto i_values = this->values;
    const auto o_values = other.values;
    for (int i = 0; i < this->dim; i++) {
      i_values[i] += a * o_values[i];
    }
  }

  void hadamard(const SparseVector<Val>& other) {
    auto i_values = this->values;
    const auto o_indices = other.indices;
    const auto o_values = other.values;
    int nnz = other.nnz;
    for (int i = 0; i < nnz; i++) {
      i_values[o_indices[i]] *= o_values[i];
    }
  }

  //dense dot_square sparse
  Val dot_square(const SparseVector<Val>& other) const {
    // assert(this->dim == other.dim);
    // int other_id = 0;
    Val dot = 0;
    auto i_values = this->values;
    const auto o_indices = other.indices;
    const auto o_values = other.values;
    int nnz = other.nnz;
    for (int i = 0; i < nnz; i++) {
      dot += (i_values[o_indices[i]] * o_values[i]) * \
        (i_values[o_indices[i]] * o_values[i]);
    }
    return dot;
  }

  void clear() override {
    std::fill(this->values, this->values + this->dim, (Val) 0);
  }

  AVector<Val>* copy() const override {
    Val *values_cp = new Val[this->dim];
    std::copy(this->values, this->values + this->dim, values_cp);
    return new DenseVector(values_cp, this->dim);
  }

  bool is_dense() const override {
    return false;
  }

  friend std::ostream& operator<<(std::ostream& os, const DenseVector<Val>& dv) {
    os << "DenseVector{" << dv.dim << ", [";
    if (dv.dim > 0) {
      os << dv.values[0];
    }
    for (int i = 1; i < dv.dim; i++) {
      os << ", " << dv.values[i];
    }
    os << "]}";
    return os;
  }
};

template <typename Val> class SparseVector : public AVector<Val> {
public:
  int *indices;
  Val *values;
  int nnz;

  SparseVector(int *inds, Val *vals, int nnz, int dim)
  : AVector<Val>(dim), indices(inds), values(vals), nnz(nnz) {
    // assert(nnz == 0 || indices[nnz - 1] < this->dim);
    ASSERT(nnz == 0 || indices[nnz - 1] <= this->dim) 
      << "indices " << indices[nnz - 1] << " is large than dim " << this->dim;
  };

  SparseVector(const SparseVector<Val>& other): AVector<Val>(other.dim), 
  indices(new int[other.nnz]), values(new Val[other.nnz]), nnz(nnz) {
    std::copy(other.indices, other.indices + other.nnz, this->indices);
    std::copy(other.values, other.values + other.nnz, this->values);
  }

  ~SparseVector() {
    if (indices != nullptr) delete[]indices;
    if (values != nullptr) delete[]values;
  }

  // sparse * sparse
  Val dot(const SparseVector<Val>& other) const override {
    // assert(this->dim == other.dim);
    Val dot = 0;
    auto i_indices = this->indices;
    auto i_values = this->values;
    int i_nnz = this->nnz;
    const auto o_indices = other.indices;
    const auto o_values = other.values;
    int o_nnz = other.nnz;
    for (int i_id = 0, o_id = 0; i_id < i_nnz && o_id < o_nnz;) {
      if (i_indices[i_id] < o_indices[o_id]) {
        i_id += 1;
      } else if (i_indices[i_id] > o_indices[o_id]) {
        o_id += 1;
      } else {
        dot += i_values[i_id] * o_values[o_id];
        i_id += 1;
        o_id += 1;
      }
    }
    return dot;
  }

  Val dot(const DenseVector<Val>& other) const override {
    // assert(this->dim == other.dim);
    Val res = 0;
    auto i_indices = this->indices;
    auto i_values = this->values;
    int i_nnz = this->nnz;
    const auto o_values = other.values;
    for (int i = 0; i < i_nnz; i++) {
      res += i_values[i] * o_values[i_indices[i]];
    }
    return res;
  }

  // this += a * other;
  // assume that *this* vector contains all non-zero elements in *other*
  void axp0(const SparseVector<Val>& other, Val a) override {
    // assert(this->dim == other.dim);
    Val dot = 0;
    auto i_indices = this->indices;
    auto i_values = this->values;
    int i_nnz = this->nnz;
    const auto o_indices = other.indices;
    const auto o_values = other.values;
    int o_nnz = other.nnz;
    for (int i_id = 0, o_id = 0; i_id < i_nnz && o_id < o_nnz;) {
      if (i_indices[i_id] < o_indices[o_id]) {
        i_id += 1;
      } else if (i_indices[i_id] > o_indices[o_id]) {
        o_id += 1;
      } else {
        // dot += i_values[i_id] * o_values[o_id];
        i_values[i_id] += a * o_values[o_id];
        i_id += 1;
        o_id += 1;
      }
    }
  }

  // this += a * other
  void axp0(const DenseVector<Val>& other, Val a) override {
    // assert(this->dim == other.dim);
    ASSERT(false) << "Unsupported operations: axp0 sparse with dense vectors";
  }

  void hadamard(SparseVector<Val> *other) {
    // assert(this->dim == other.dim);
    Val dot = 0;
    auto i_indices = this->indices;
    auto i_values = this->values;
    int i_nnz = this->nnz;
    const auto o_indices = other.indices;
    const auto o_values = other.values;
    int o_nnz = other.nnz;
    for (int i_id = 0, o_id = 0; i_id < i_nnz && o_id < o_nnz;) {
      if (i_indices[i_id] < o_indices[o_id]) {
        i_id += 1;
      } else if (i_indices[i_id] > o_indices[o_id]) {
        o_id += 1;
      } else {
        i_values[i_id] *= o_values[o_id];
        i_id += 1;
        o_id += 1;
      }
    }
  }

  void hadamard(const DenseVector<Val>& other){
    // assert(this->dim == other.dim);
    auto i_indices = this->indices;
    auto i_values = this->values;
    int i_nnz = this->nnz;
    const auto o_values = other.values;
    for (int i = 0; i < i_nnz; i++) {
      i_values[i] *= o_values[i_indices[i]];
    }
  }

  // sparse * sparse
  Val dot_square(const SparseVector<Val>& other) {
    // assert(this->dim == other.dim);
    Val dot = 0;
    auto i_indices = this->indices;
    auto i_values = this->values;
    int i_nnz = this->nnz;
    const auto o_indices = other.indices;
    const auto o_values = other.values;
    int o_nnz = other.nnz;
    for (int i_id = 0, o_id = 0; i_id < i_nnz && o_id < o_nnz;) {
      if (i_indices[i_id] < o_indices[o_id]) {
        i_id += 1;
      } else if (i_indices[i_id] > o_indices[o_id]) {
        o_id += 1;
      } else {
        dot += (i_values[i_id] * o_values[o_id]) * \
          (i_values[i_id] * o_values[o_id]);
        dot += i_values[i_id] * o_values[o_id];
        i_id += 1;
        o_id += 1;
      }
    }
    return dot;
  }

  void clear() override {
    std::fill(this->values, this->values + this->nnz, (Val) 0);
  }

  AVector<Val>* copy() const override {
    int *indices_cp = new int[this->nnz];
    std::copy(this->indices, this->indices + this->nnz, indices_cp);
    Val *values_cp = new Val[this->nnz];
    std::copy(this->values, this->values + this->nnz, values_cp);
    return new SparseVector(indices_cp, values_cp, this->nnz, this->dim);
  }

  bool is_dense() const override {
    return false;
  }

  friend std::ostream& operator<<(std::ostream& os, const SparseVector<Val>& sv) {
    os << "SparseVector{" << sv.dim << ", " << sv.nnz << ", [";
    if (sv.nnz > 0) {
      os << '(' << sv.indices[0] << ", " << sv.values[0] << ')';
    }
    for (int i = 1; i < sv.nnz; i++) {
      os << ", (" << sv.indices[i] << ", " << sv.values[i] << ')';
    }
    os << "]}";
    return os;
  }
};

#endif // __HETU_ML_DATA_VECTORS_H_
