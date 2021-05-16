#ifndef __HETU_ML_OBJECTIVE_LOSS_H_
#define __HETU_ML_OBJECTIVE_LOSS_H_

#include "common/logging.h"
#include "common/math.h"
// #include <stdio.h>

namespace hetu { 
namespace ml {

class LossFactory;

// for binary classes, we assume the labels are in {-1, +1}
template <typename Val>
class Loss {
public:
  virtual Val loss(Val predict, Val label) const = 0;
  virtual Val grad(Val predict, Val label) const = 0;
  virtual const std::string& name() const = 0;
protected:
  Loss() {}
};

/**
 * \brief Logistic Loss.
 * loss = log(1 + exp(−y * y'))
 * grad = -y / (1 + exp(y * y'))
 */
template <typename Val>
class LogisticLoss : public Loss<Val> {
public:
  friend class LossFactory;
  inline Val loss(Val predict, Val label) const override {
    return log(1 + exp(-label * predict));
  }
  inline Val grad(Val predict, Val label) const override {
    return -label / (1 + exp(predict * label));
  }
  inline const std::string& name() const override { return NAME; }
  static const std::string NAME;
private:
  LogisticLoss() {}
  static const LogisticLoss<Val> INSTANCE;
};

/**
 * \brief Hinge loss.
 * loss = max(0,1 − y * y')
 * grad = 0 if (1 - y * y') < 0
 * grad = -y if (1 - y * y') > 0
 */
template <typename Val>
class HingeLoss : public Loss<Val> {
public:
  friend class LossFactory;
  inline Val loss(Val predict, Val label) const override {
    if (predict * label < 1) {
      return 1 - predict * label;
    } else {
      return 0;
    }
  }
  inline Val grad(Val predict, Val label) const override {
    if (predict * label < 1) {
      return -label;
    } else {
      return 0;
    }
  }
  inline const std::string& name() const override { return NAME; }
  static const std::string NAME;
private:
  HingeLoss() {}
  static const HingeLoss<Val> INSTANCE;
};

/**
 * \brief Square loss.
 * loss = 1/2 * (y - y')^2
 * grad = y' - y
 */
template <typename Val>
class SquareLoss : public Loss<Val> {
public:
  friend class LossFactory;
  inline Val loss(Val predict, Val label) const override {
    Val loss = 0.5 * (predict - label) * (predict - label);
    return loss;
  }
  inline Val grad(Val predict, Val label) const override {
    const Val threshold = 20; // to avoid overflow
    if (predict > threshold) {
      predict = threshold;
    } else if (predict < -threshold) {
      predict = -threshold;
    }
    return predict - label;
  }
  inline const std::string& name() const override { return NAME; }
  static const std::string NAME;
private:
  SquareLoss() {}
  static const SquareLoss<Val> INSTANCE;
};

template<typename P, typename L, typename G>
class RichLoss {
public:
  virtual ~RichLoss() {}
  
  virtual G FirstOrderGrad(P pred, L label) const = 0;

  virtual G SecondOrderGrad(P pred, L label) const = 0;

  virtual G SecondOrderGrad(P pred, L label, G first_grad) const = 0;

  virtual void FirstOrderGrad(const P* pred, L label, 
                              int num_classes, G* ret) const = 0;

  virtual void SecondOrderGradDiag(const P* pred, L label, 
                                   int num_classes, G* ret) const = 0;

  virtual void SecondOrderGradDiag(const P* pred, L label, 
                                   G* first_grad, int num_classes, 
                                   G* ret) const = 0;
  
  virtual void SecondOrderGradFull(const P* pred, L label, 
                                   int num_classes, G* ret) const = 0;

  virtual void SecondOrderGradFull(const P* pred, L label, 
                                   G* first_grad, int num_classes, 
                                   G* ret) const = 0;

  virtual const std::string& name() const = 0;
};

template<typename P, typename L, typename G>
class RichSquareLoss : public RichLoss<P, L, G> {
public:
  friend class LossFactory;

  RichSquareLoss(RichSquareLoss const&) = delete;
  
  void operator=(RichSquareLoss const&) = delete;

  inline G FirstOrderGrad(P pred, L label) const override {
    return pred - label;
  }

  ~RichSquareLoss() {}

  inline G SecondOrderGrad(P pred, L label) const override {
    return 1;
  }

  inline G SecondOrderGrad(P pred, L label, G first_grad) const override {
    return 1;
  }

  inline void FirstOrderGrad(const P* pred, L label, 
                             int num_classes, G* ret) const override {
    int true_label = static_cast<int>(label);
    for (int i = 0; i < num_classes; i++)
      ret[i] = pred[i] - ((true_label == i) ? 1 : 0);
  }

  inline void SecondOrderGradDiag(const P* pred, L label, 
                                  int num_classes, G* ret) const override {
    for (int i = 0; i < num_classes; i++)
      ret[i] = 1;
  }

  inline void SecondOrderGradDiag(const P* pred, L label, 
                                  G* first_grad, int num_classes, 
                                  G* ret) const override {
    for (int i = 0; i < num_classes; i++)
      ret[i] = 1;
  }

  inline void SecondOrderGradFull(const P* pred, L label, 
                                  int num_classes, G* ret) const override {
    int row_offset = 0;
    for (int i = 0; i < num_classes; i++) {
      for (int j = 0; j < i; j++)
        ret[row_offset + j] = 0;
      ret[row_offset + i] = 1;
      row_offset += i + 1;
    }
  }

  inline void SecondOrderGradFull(const P* pred, L label, 
                                  G* first_grad, int num_classes, 
                                  G* ret) const override {
    SecondOrderGradFull(pred, label, num_classes, ret);
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  RichSquareLoss() {}
  static const RichSquareLoss<P, L, G> INSTANCE;
};

template<typename P, typename L, typename G>
class RichLogisticLoss : public RichLoss<P, L, G> {
public:
  friend class LossFactory;

  RichLogisticLoss(RichLogisticLoss const&) = delete;
  
  void operator=(RichLogisticLoss const&) = delete;

  ~RichLogisticLoss() {}

  inline G FirstOrderGrad(P pred, L label) const override {
    G prob = Sigmoid<G>(pred);
    return prob - label;
  }

  inline G SecondOrderGrad(P pred, L label) const override {
    G prob = Sigmoid<G>(pred);
    return MAX(prob * (1 - prob), EPSILON);
  }

  inline G SecondOrderGrad(P pred, L label, G first_grad) const override {
    G prob = static_cast<G>(first_grad + label);
    return MAX(prob * (1 - prob), EPSILON);
  }

  inline void FirstOrderGrad(const P* pred, L label, 
                             int num_classes, G* ret) const override {
    Softmax<P, G>(pred, num_classes, ret);
    int true_label = static_cast<int>(label);
    for (int i = 0; i < num_classes; i++)
      ret[i] = (true_label == i) ? (ret[i] - 1) : ret[i];
  }

  inline void SecondOrderGradDiag(const P* pred, L label, 
                                  int num_classes, G* ret) const override {
    Softmax<P, G>(pred, num_classes, ret);
    for (int i = 0; i < num_classes; i++)
      ret[i] = MAX(ret[i] * (1 - ret[i]), EPSILON);
  }

  inline void SecondOrderGradDiag(const P* pred, L label, 
                                  G* first_grad, int num_classes, 
                                  G* ret) const override {
    int true_label = static_cast<int>(label);
    for (int i = 0; i < num_classes; i++) {
      G prob = (true_label == i) ? (first_grad[i] + 1) : first_grad[i];
      ret[i] = MAX(prob * (1 - prob), EPSILON);
    }
  }

  inline void SecondOrderGradFull(const P* pred, L label, 
                                  int num_classes, G* ret) const override {
    std::vector<G> prob(num_classes);
    Softmax<P, G>(pred, num_classes, prob.data());
    int row_offset = 0;
    for (int i = 0; i < num_classes; i++) {
      for (int j = 0; j < i; j++)
        ret[row_offset + j] = -MAX(prob[i] * prob[j], EPSILON);
      ret[row_offset + i] = MAX(prob[i] * (1 - prob[i]), EPSILON);
      row_offset += i + 1;
    }
  }

  inline void SecondOrderGradFull(const P* pred, L label, 
                                  G* first_grad, int num_classes, 
                                  G* ret) const override {
    int true_label = static_cast<int>(label);
    std::vector<G> prob(num_classes);
    for (int i = 0; i < num_classes; i++) 
      prob[i] = (true_label == i) ? (first_grad[i] + 1) : first_grad[i];
    int row_offset = 0;
    for (int i = 0; i < num_classes; i++) {
      for (int j = 0; j < i; j++)
        ret[row_offset + j] = -MAX(prob[i] * prob[j], EPSILON);
      ret[row_offset + i] = MAX(prob[i] * (1 - prob[i]), EPSILON);
      row_offset += i + 1;
    }
  }

  inline const std::string& name() const { return NAME; }
  static const std::string NAME;
private:
  RichLogisticLoss() {}
  static const RichLogisticLoss<P, L, G> INSTANCE;
};

template<typename Val>
const LogisticLoss<Val> LogisticLoss<Val>::INSTANCE;
template<typename Val>
const std::string LogisticLoss<Val>::NAME = "logistic";
template<typename Val>
const HingeLoss<Val> HingeLoss<Val>::INSTANCE;
template<typename Val>
const std::string HingeLoss<Val>::NAME = "hinge";
template<typename Val>
const SquareLoss<Val> SquareLoss<Val>::INSTANCE;
template<typename Val>
const std::string SquareLoss<Val>::NAME = "square";
template<typename P, typename L, typename G>
const std::string RichSquareLoss<P, L, G>::NAME = "square";
template<typename P, typename L, typename G>
const RichSquareLoss<P, L, G> RichSquareLoss<P, L, G>::INSTANCE;
template<typename P, typename L, typename G>
const std::string RichLogisticLoss<P, L, G>::NAME = "logistic";
template<typename P, typename L, typename G>
const RichLogisticLoss<P, L, G> RichLogisticLoss<P, L, G>::INSTANCE;

class LossFactory {
public:
  template<typename Val>
  static const Loss<Val>* GetLoss(const std::string& loss) {
    if (!loss.compare(LogisticLoss<Val>::NAME)) { 
      return &LogisticLoss<Val>::INSTANCE;
    } else if (!loss.compare(HingeLoss<Val>::NAME)) { 
      return &HingeLoss<Val>::INSTANCE;
    } else if (!loss.compare(SquareLoss<Val>::NAME)) {
      return &SquareLoss<Val>::INSTANCE;
    } else {
      HML_LOG_FATAL << "No such loss: " << loss;
      return nullptr;
    }
  }

  template<typename P, typename L, typename G>
  static const RichLoss<P, L, G>* GetRichLoss(const std::string& loss) {
    if (!loss.compare(RichLogisticLoss<P, L, G>::NAME)) { 
      return &RichLogisticLoss<P, L, G>::INSTANCE;
    } else if (!loss.compare(RichSquareLoss<P, L, G>::NAME)) {
      return &RichSquareLoss<P, L, G>::INSTANCE;
    } else {
      HML_LOG_FATAL << "No such rich loss: " << loss;
      return nullptr;
    }
  }
};

} // namespace ml
} // namespace hetu

#endif // __HETU_ML_OBJECTIVE_LOSS_H_
