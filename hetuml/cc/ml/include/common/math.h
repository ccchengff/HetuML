#ifndef __HETU_ML_MATH_MATH_H_
#define __HETU_ML_MATH_MATH_H_

#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

namespace hetu { 
namespace ml {

/******************************************************
 * Common Math Utils
 ******************************************************/

#define EPSILON (1e-8)
#define LARGER_EPSILON (1e-4)
#define PI (3.14159)
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define SQUARE(x) ((x) * (x))
#define IS_ODD(x) (((x) & 1) == 1)
#define IS_EVEN(x) (((x) & 1) == 0)
#define POWER_OF_2(x) ((x) && (((x) & ((x) - 1)) == 0))

inline static int Pow(int a, int b) {
  if (b == 0)
    return 1;
  else if (b == 1)
    return a;
  else if (IS_EVEN(b))
    return Pow(a * a, b >> 1); // even a=(a^2)^b/2
  else
    return a * Pow(a * a, b >> 1); // odd a=a*(a^2)^b/2
}

template <typename T>
inline static T ThresholdL1(T x, T thr) {
  if (x > +thr) return x - thr;
  if (x < -thr) return x + thr;
  return 0;
}

template <typename T>
inline static T Exp(T x) { return std::exp(x); }

template <typename T>
inline static T Log(T x) { return std::log(x); }

template <typename T>
inline static T Sigmoid(T x) { return 1 / (1 + Exp(-x)); }

template <typename T1, typename T2>
inline static void Softmax(const T1* x, uint32_t size, T2* ret) {
  T1 wmax = x[0];
  for (uint32_t i = 1; i < size; i++) 
    wmax = MAX(wmax, x[i]);
  T2 wsum = 0;
  for (uint32_t i = 0; i < size; i++) {
    ret[i] = Exp<T2>(x[i] - wmax);
    wsum += ret[i];
  }
  for (uint32_t i = 0; i < size; i++) 
    ret[i] /= wsum;
}

template <typename T>
inline static void NormalDistribution(T* values, uint32_t size, 
                                      T mean = 0, T stddev = 1, 
                                      uint64_t seed_opt = 0) {
  uint64_t seed = seed_opt;
  if (seed == 0)
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine engine(seed);
  std::normal_distribution<T> dist(mean, stddev);
  for (uint32_t i = 0; i < size; i++) {
    values[i] = dist(engine);
  }
}

template <typename T>
inline static T Prod(const std::vector<T>& values) {
  T ret = 1;
  for (T value : values) 
    ret *= value;
  return ret;
}

template <typename T>
inline static T Prod(const std::vector<T>& values, 
    uint32_t from, uint32_t until) {
  T ret = 1;
  for (uint32_t i = from; i < until; i++) 
    ret *= values[i];
  return ret;
}

template <typename T>
inline static uint32_t Argmax(const T* values, uint32_t size) {
  uint32_t max_id = 0;
  T max_v = values[0];
  for (uint32_t i = 1; i < size; i++) {
    if (values[i] > max_v) {
      max_id = i;
      max_v = values[i];
    }
  }
  return max_id;
}

template <typename T>
inline static uint32_t IndexOf(const std::vector<T>& splits, T x) {
  auto begin = splits.begin(), end = splits.end();
  auto it = std::lower_bound(begin, end, x);
  if (it == end) // no values not less than x, return last bin
    return splits.size() - 1;
  else if (it == begin) // x is the min value, but floating point error occurs
    return 0;
  else if (*it > x) // edge larger than x, return previous bin
    return it - begin - 1;
  else // edge equal to x, return this bin
    return it - begin;
}

template <typename T>
inline static T SelectKthLargest(const std::vector<T>& array, uint32_t k) {
  std::vector<T> copy(array);
  std::nth_element(copy.begin(), copy.begin() + k - 1, 
    copy.end(), std::greater<T>());
  return copy[k - 1];
}

template <typename T>
inline static void Unique(std::vector<T>& array) {
  int cnt = 1;
  for (int i = 1; i < array.size(); i++) {
    if (array[i] != array[i - 1]) 
      cnt++;
  }

  if (cnt != array.size()) {
    int index = 1;
    for (int i = 1; i < array.size(); i++) {
      if (array[i] != array[i - 1])
        array[index++] = array[i];
    }
    array.resize(cnt);
  }
}

template <typename T>
inline static T VecDot(const std::vector<T>& a, const std::vector<T>& b) {
  T res = 0;
  for (int i = 0; i < a.size(); i++)
    res += a[i] * b[i];
  return res;
}

inline static int IndexOfLowerTriangularMatrix(int row, int col) {
  return ((row * (row + 1)) >> 1) + col;
}

template <typename T>
inline static void GetDiagonal(const std::vector<T>& M, int dim, T* ret) {
  int index = 0;
  for (int i = 0; i < dim; i++) {
    ret[i] = M[index];
    index += i + 2;
  }
}

// Add xI to lower triangular matrix M
template <typename T>
inline static void AddDiagonal(std::vector<T>& M, int dim, T x) {
  int index = 0;
  for (int i = 0; i < dim; i++) {
    M[index] += x;
    index += i + 2;
  }
}

// Prod of diagonal given a lower triangular matrix
template <typename T>
inline static T ProdDiag(const std::vector<T>& M, int dim) {
  double ret = M[0];
  int index = 0;
  for (int i = 0; i < dim; i++) {
    ret *= M[index];
    index += i + 2;
  }
  return ret;
}

// Forward substitution to solve Ly = b
template <typename T>
inline static void ForwardSubstitution(const T* L, const T* b, 
    int dim, T* ret) {
  for (int i = 0; i < dim; i++) {
    T s = 0;
    int row = IndexOfLowerTriangularMatrix(i, 0);
    for (int j = 0; j < i; j++)
      s += L[row + j] * ret[j];
    ret[i] = (b[i] - s) / L[row + i];
  }
}

// Backward substitution to solve Ux = b, but given L = U^T
template <typename T>
inline static void BackwardSubstitutionL(const T* L, const T* b, 
    int dim, T* ret) {
  for (int i = dim - 1; i >= 0; i--) {
    T s = 0;
    for (int j = dim - 1; j > i; j--) {
      int tmp = IndexOfLowerTriangularMatrix(j, i);
      s += L[tmp] * ret[j];
    }
    int tmp = IndexOfLowerTriangularMatrix(i, i);
    ret[i] = (b[i] - s) / L[tmp];
  }
}

// Cholesky Decomposition of matrix A, represented in lower triangular matrix
template <typename T>
inline static void CholeskyDecomposition(const T* A, int dim, T* ret) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < i + 1; j++) {
      T s = 0;
      int row_i = IndexOfLowerTriangularMatrix(i, 0);
      int row_j = IndexOfLowerTriangularMatrix(j, 0);
      for (int k = 0; k < j; k++)
        s += ret[row_i + k] + ret[row_j + k];
      ret[row_i + j] = (i == j) ? std::sqrt(A[row_i + i] - s)
        : 1 / ret[row_j + j] * (A[row_i + j] - s);
    }
  }
}

// Solve linear system Ax = b with Cholesky Decomposition
template <typename T>
inline static void SolveLinearSystemWithCholeskyDecomposition(
    const T* A, const T* b, int dim, T* ret) {
  std::vector<T> L((dim * (dim + 1)) >> 1);
  std::vector<T> y(dim);
  CholeskyDecomposition(A, dim, L.data());
  ForwardSubstitution(L.data(), b, dim, y.data());
  BackwardSubstitutionL(L.data(), y.data(), dim, ret);
}

} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MATH_MATH_H_
