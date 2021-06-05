#ifndef __HETU_ML_GBDT_OBJECTIVE_H
#define __HETU_ML_GBDT_OBJECTIVE_H

#include "common/logging.h"
#include "common/math.h"
// #include <string>
// #include <algorithm>

namespace hetu { 
namespace ml {

class MetricFactory;

template<typename P, typename L, typename M>
class EvalMetric {
public:
  virtual ~EvalMetric() {}

  virtual M EvalBinary(const P* preds, const L* labels, uint32_t size) const {
    HML_LOG_FATAL << "Metric[" << this->name() << "] does not support "
      << "scalar-label tasks";
    return std::numeric_limits<M>::quiet_NaN();
  }

  virtual M EvalMulti(const P* preds, const L* labels, 
                      int num_classes, uint32_t size) const {
    HML_LOG_FATAL << "Metric[" << this->name() << "] does not support "
      << "multi-label tasks";
    return std::numeric_limits<M>::quiet_NaN();
  }

  virtual const std::string& name() const = 0;
};

template<typename P, typename L, typename M>
class AverageEvalMetric : public EvalMetric<P, L, M> {
public:
  virtual ~AverageEvalMetric() {}

  M EvalBinary(const P* preds, const L* labels, uint32_t size) const {
    return EvalBinary(preds, labels, 0, size);
  }

  M EvalBinary(const P* preds, const L* labels, 
               uint32_t from, uint32_t until) const {
    M ret = 0;
    #pragma omp parallel for reduction(+:ret)
    for (uint32_t i = from; i < until; i++) 
      ret += EvalBinaryOne(preds[i], labels[i]);
    return ret / (until - from);
  }

  virtual M EvalBinaryOne(const P pred, const L label) const {
    HML_LOG_FATAL << "Metric[" << this->name() << "] does not support "
      << "scalar-label tasks";
    return std::numeric_limits<M>::quiet_NaN();
  }

  M EvalMulti(const P* preds, const L* labels, 
              int num_classes, uint32_t size) const {
    return EvalMulti(preds, labels, num_classes, 0, size);
  }

  M EvalMulti(const P* preds, const L* labels, 
              int num_classes, uint32_t from, uint32_t until) const {
    ASSERT(from <= until) << "Invalid range: [" << from << ", " ")";
    M ret = 0;
    #pragma omp parallel for reduction(+:ret)
    for (uint32_t i = from; i < until; i++) 
      ret += EvalMultiOne(preds + i * num_classes, labels[i], num_classes);
    return ret / (until - from);
  }

  virtual M EvalMultiOne(const P* pred, const L label, int num_classes) const {
    HML_LOG_FATAL << "Metric[" << this->name() << "] does not support "
      << "multi-label tasks";
    return std::numeric_limits<M>::quiet_NaN();
  }
};

template<typename P, typename L, typename M>
class MSEMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  MSEMetric(MSEMetric const&) = delete;
  
  void operator=(MSEMetric const&) = delete;

  ~MSEMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    M diff = pred - label;
    return diff * diff;
  }

  M EvalMultiOne(const P* pred, const L label, 
      int num_classes) const {
    M ret = 0;
    int true_label = static_cast<int>(label);
    for (int k = 0; k < num_classes; k++) {
      M diff = pred[k] - (k == true_label ? 1 : 0);
      ret += diff * diff;
    }
    return ret;
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  MSEMetric() {}
  static const MSEMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class ErrorMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  ErrorMetric(ErrorMetric const&) = delete;
  
  void operator=(ErrorMetric const&) = delete;

  ~ErrorMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    return pred >= 0.0f ? 1 - label : label;
  }

  M EvalMultiOne(const P* pred, const L label, 
      int num_classes) const {
    int label_int = static_cast<int>(label);
    return Argmax<P>(pred, num_classes) != label_int ? 1 : 0;
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  ErrorMetric() {}
  static const ErrorMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class PrecisionMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  PrecisionMetric(PrecisionMetric const&) = delete;
  
  void operator=(PrecisionMetric const&) = delete;

  ~PrecisionMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    return pred >= 0.0f ? label : 1 - label;
  }

  M EvalMultiOne(const P* pred, const L label, 
      int num_classes) const {
    int label_int = static_cast<int>(label);
    return Argmax<P>(pred, num_classes) == label_int ? 1 : 0;
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  PrecisionMetric() {}
  static const PrecisionMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class LogLossMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  LogLossMetric(LogLossMetric const&) = delete;
  
  void operator=(LogLossMetric const&) = delete;

  ~LogLossMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    P prob = Sigmoid<P>(pred);
    int label_int = static_cast<int>(label);
    return (label_int == 1) ? -Log<M>(MAX(prob, LARGER_EPSILON)) 
      : -Log<M>(MAX(1 - prob, LARGER_EPSILON));
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  LogLossMetric() {}
  static const LogLossMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class CrossEntropyMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  CrossEntropyMetric(CrossEntropyMetric const&) = delete;
  
  void operator=(CrossEntropyMetric const&) = delete;

  ~CrossEntropyMetric() {}

  M EvalMultiOne(const P* pred, const L label, 
      int num_classes) const {
    M sum = 0;
    for (int k = 0; k < num_classes; k++) 
      sum += Exp<M>(pred[k]);
    int label_int = static_cast<int>(label);
    M prob = Exp<M>(pred[label_int]) / sum;
    return -Log<M>(MAX(prob, LARGER_EPSILON));
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  CrossEntropyMetric() {}
  static const CrossEntropyMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class NegYErrorMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  NegYErrorMetric(NegYErrorMetric const&) = delete;
  
  void operator=(NegYErrorMetric const&) = delete;

  ~NegYErrorMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    return pred * label >= 0 ? 0 : 1;
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  NegYErrorMetric() {}
  static const NegYErrorMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class NegYPrecisionMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  NegYPrecisionMetric(NegYPrecisionMetric const&) = delete;
  
  void operator=(NegYPrecisionMetric const&) = delete;

  ~NegYPrecisionMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    return pred * label >= 0 ? 1 : 0;
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  NegYPrecisionMetric() {}
  static const NegYPrecisionMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class NegYLogLossMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  NegYLogLossMetric(NegYLogLossMetric const&) = delete;
  
  void operator=(NegYLogLossMetric const&) = delete;

  ~NegYLogLossMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    return log(1 + exp(-label * pred));
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  NegYLogLossMetric() {}
  static const NegYLogLossMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
class HingeLossMetric : public AverageEvalMetric<P, L, M> {
public:
  friend class MetricFactory;

  HingeLossMetric(HingeLossMetric const&) = delete;
  
  void operator=(HingeLossMetric const&) = delete;

  ~HingeLossMetric() {}

  M EvalBinaryOne(const P pred, const L label) const {
    if (pred * label < 1) {
      return 1 - pred * label;
    } else {
      return 0;
    }
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  HingeLossMetric() {}
  static const HingeLossMetric<P, L, M> INSTANCE;
};

template<typename P, typename L, typename M>
const MSEMetric<P, L, M> MSEMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string MSEMetric<P, L, M>::NAME = "mse";
template<typename P, typename L, typename M>
const ErrorMetric<P, L, M> ErrorMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string ErrorMetric<P, L, M>::NAME = "error";
template<typename P, typename L, typename M>
const PrecisionMetric<P, L, M> PrecisionMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string PrecisionMetric<P, L, M>::NAME = "precision";
template<typename P, typename L, typename M>
const LogLossMetric<P, L, M> LogLossMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string LogLossMetric<P, L, M>::NAME = "log-loss";
template<typename P, typename L, typename M>
const CrossEntropyMetric<P, L, M> CrossEntropyMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string CrossEntropyMetric<P, L, M>::NAME = "cross-entropy";
template<typename P, typename L, typename M>
const NegYErrorMetric<P, L, M> NegYErrorMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string NegYErrorMetric<P, L, M>::NAME = \
  ErrorMetric<P, L, M>::NAME;
template<typename P, typename L, typename M>
const NegYPrecisionMetric<P, L, M> NegYPrecisionMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string NegYPrecisionMetric<P, L, M>::NAME = \
  PrecisionMetric<P, L, M>::NAME;
template<typename P, typename L, typename M>
const NegYLogLossMetric<P, L, M> NegYLogLossMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string NegYLogLossMetric<P, L, M>::NAME = \
  LogLossMetric<P, L, M>::NAME;
template<typename P, typename L, typename M>
const HingeLossMetric<P, L, M> HingeLossMetric<P, L, M>::INSTANCE;
template<typename P, typename L, typename M>
const std::string HingeLossMetric<P, L, M>::NAME = "hinge-loss";

class MetricFactory {
public:
  template <typename P, typename L, typename M>
  static const EvalMetric<P, L, M>*
  GetEvalMetric(const std::string& metric, bool neg_y = false) {
    if (!metric.compare(MSEMetric<P, L, M>::NAME)) {
      return &MSEMetric<P, L, M>::INSTANCE;
    } else if (!metric.compare(ErrorMetric<P, L, M>::NAME)) {
      if (!neg_y) {
        return &ErrorMetric<P, L, M>::INSTANCE;
      } else {
        return &NegYErrorMetric<P, L, M>::INSTANCE;
      }
    } else if (!metric.compare(PrecisionMetric<P, L, M>::NAME)) {
      if (!neg_y) {
        return &PrecisionMetric<P, L, M>::INSTANCE;
      } else {
        return &NegYPrecisionMetric<P, L, M>::INSTANCE;
      }
    } else if (!metric.compare(LogLossMetric<P, L, M>::NAME)) {
      if (!neg_y) {
        return &LogLossMetric<P, L, M>::INSTANCE;
      } else {
        return &NegYLogLossMetric<P, L, M>::INSTANCE;
      }
    } else if (!metric.compare(CrossEntropyMetric<P, L, M>::NAME)) {
      return &CrossEntropyMetric<P, L, M>::INSTANCE;
    } else if (!metric.compare(HingeLossMetric<P, L, M>::NAME)) {
      return &HingeLossMetric<P, L, M>::INSTANCE;
    } else {
      HML_LOG_FATAL << "No such metric: " << metric;
      return nullptr;
    }
  }

  template <typename M>
  static std::string to_string(const std::vector<std::string>& names, 
                               const std::vector<M>& metrics) {
    ASSERT_EQ(names.size(), metrics.size()) << "Size mismatch: " 
      << names.size() << " vs. " << metrics.size();
    if (names.size() == 0) 
      return std::string();
    std::ostringstream oss;
    oss << names[0] << '[' << metrics[0] << ']';
    for (size_t i = 1; i < names.size(); i++)
      oss << ' ' << names[i] << '[' << metrics[i] << ']';
    return oss.str();
  }
};

} // namespace ml
} // namespace hetu

#endif
