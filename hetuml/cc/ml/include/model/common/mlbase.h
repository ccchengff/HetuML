#ifndef __HETU_ML_MODEL_COMMON_MLBASE_H_
#define __HETU_ML_MODEL_COMMON_MLBASE_H_

#include "model/common/argparse.h"
#include "data/dataset.h"
#include "objective/loss.h"
#include "objective/metric.h"
#include "common/math.h"
#include <fstream>

namespace hetu { 
namespace ml {

class MLParam {
public:
  MLParam(const Args& args = {}, const Args& default_args = {}) {
    this->all_args = default_args;
    for (const auto& it : args) {
      if (this->all_args.find(it.first) != this->all_args.end()) {
        HML_LOG_DEBUG << "Parsing param key[" << it.first 
          << "] value[" << it.second << "]";
        this->all_args[it.first] = it.second;
      } else {
        HML_LOG_DEBUG << "Skipping param key[" << it.first 
          << "] value[" << it.second << "]";
      }
    }
  }
  template <typename T>
  inline T get(const std::string& key) const {
    return argparse::Get<T>(this->all_args, key);
  }
  inline bool get_bool(const std::string& key) const {
    return argparse::GetBool(this->all_args, key);
  }
  inline const Args& get_all_args() const { return this->all_args; }
  friend std::ostream& operator<<(std::ostream& os, const MLParam& param) {
    const auto& all_keys = param.keys();
    for (const auto& key : all_keys) {
      os << "|" << key << " = " << param.all_args.at(key) << std::endl;
    }
    return os;
  }
  friend std::istream& operator>>(std::istream& is, MLParam& param) {
    const auto& all_keys = param.keys();
    std::string buf;
    for (const auto& key : all_keys) {
      std::getline(is, buf);
      std::string prefix = "|" + key + " = ";
      ASSERT(buf.find(prefix) == 0) 
        << "Cannot find key[" << key << "] in line: " << buf;
      size_t offset = prefix.length();
      std::string value = buf.substr(prefix.length());
      param.all_args[key] = value;
    }
    param.InitAndCheckParam();
    return is;
  }
  virtual void InitAndCheckParam() = 0;
  virtual const std::vector<std::string> keys() const = 0;
protected:
  Args all_args;
};

template <typename Param> 
struct is_ml_param { 
  static const bool value = std::is_base_of<MLParam, Param>::value;
};

template <typename Param>
class MLBase {
public:
  MLBase(const Args& args = {}) {
    static_assert(is_ml_param<Param>::value, 
      "The template class is not derived from MLParam");
    this->params.reset(new Param(args));
  }
  inline void LoadModel(const std::string& path) {
    std::ifstream is(path, std::ifstream::in);
    ASSERT(is) << "Cannot open path " << path << " to load model";
    is >> *this->params;
    this->LoadFromStream(is);
    is.close();
    HML_LOG_INFO << "Load " << this->name() 
      << " model from " << path << " done";
  }
  void SaveModel(const std::string& path) {
    std::ofstream os(path, std::ofstream::out);
    ASSERT(os) << "Cannot open path " << path << " to save model";
    os << *this->params;
    this->DumpToStream(os);
    os.flush();
    os.close();
    HML_LOG_INFO << "Save " << this->name() 
      << " model to " << path << " done";
  }
  virtual const char* name() const = 0;
  const Param& get_params() const { return *this->params; }
protected:
  virtual void LoadFromStream(std::istream& is) = 0;
  virtual void DumpToStream(std::ostream& os) = 0;
  std::unique_ptr<Param> params;
};

typedef float label_t;

template<typename Val, typename Param>
class SupervisedMLBase : public MLBase<Param> {
public:
  SupervisedMLBase(const Args& args = {}): MLBase<Param>(args) {}
  
  virtual void 
  Fit(const Dataset<label_t, Val>& train_data, 
      const Dataset<label_t, Val>& valid_data = {}) = 0;
  
  virtual void 
  Predict(std::vector<label_t>& ret, const DataMatrix<Val>& features) {
    this->Predict(ret, features, 0, features.get_num_instances());
  }
  
  virtual void 
  Predict(std::vector<label_t>& ret, const DataMatrix<Val>& features, 
          size_t start_id, size_t end_id) = 0;
  
  inline virtual std::vector<Val> 
  Evaluate(const Dataset<label_t, Val>& eval_data, 
           const std::vector<std::string>& metrics) {
    std::vector<Val> res;
    res.reserve(metrics.size());
    auto num_eval = eval_data.get_num_instances();
    if (num_eval > 0) {
      std::vector<label_t> preds;
      const auto& labels = eval_data.get_labels();
      this->Predict(preds, eval_data);
      for (const auto& metric_name : metrics) {
        const auto* eval_metric = \
          MetricFactory::GetEvalMetric<Val, label_t, Val>(
            metric_name, this->use_neg_y());
        Val metric = preds.size() == num_eval \
          ? eval_metric->EvalBinary(preds.data(), labels.data(), num_eval) \
          : eval_metric->EvalMulti(preds.data(), labels.data(), 
                                   this->get_num_label(), num_eval);
        res.push_back(metric);
      }
    } else {
      res.resize(metrics.size(), (Val) 0);
    }
    return std::move(res);
  }

  inline virtual int get_num_label() const { return 2; }

  // whether the algorithm uses +1/-1 labels for binary classification
  inline virtual bool use_neg_y() const { return false; }
};

} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_COMMON_MLBASE_H_
