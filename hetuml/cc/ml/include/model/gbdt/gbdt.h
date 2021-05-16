#ifndef __HETU_ML_MODEL_GBDT_GBDT_H_
#define __HETU_ML_MODEL_GBDT_GBDT_H_

#include "model/common/mlbase.h"
#include "model/gbdt/model.h"
#include "model/gbdt/train/trainer.h"

namespace hetu { 
namespace ml {
namespace gbdt {

template <typename Val>
class GBDT : public SupervisedMLBase<Val, GBDTParam> {
public:
  inline GBDT(const Args& args = {}): SupervisedMLBase<Val, GBDTParam>(args) {}

  inline ~GBDT() {}

  inline void Fit(const Dataset<label_t, Val>& train_data, 
                  const Dataset<label_t, Val>& valid_data = {}) override {
    TIK(fit);
    HML_LOG_INFO << "Start to fit " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;

    GBDTTrainer trainer(*this->params);
    auto* model = trainer.Fit(train_data, valid_data);
    this->model.reset(model);
    
    TOK(fit);
    HML_LOG_INFO << "Fit " << this->name() << " model"
      << " cost " << COST_MSEC(fit) << " ms";
  }

  inline void Predict(std::vector<label_t>& ret, 
                      const DataMatrix<Val>& features, 
                      size_t start_id, size_t end_id) override {
    ASSERT(!this->is_empty()) << "Model is empty";
    auto pred_size = this->model->get_param().pred_size();
    ret.resize((end_id - start_id) * pred_size);
    if (features.is_dense()) {
      if (pred_size == 1) {
        for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
          const auto& dv = features.get_dense_feature(ins_id);
          ret[ins_id - start_id] = this->model->predict_scalar(dv);
        }
      } else {
        for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
          const auto& dv = features.get_dense_feature(ins_id);
          float* const preds = ret.data() + (ins_id - start_id) * pred_size;
          this->model->predict_vector(dv, preds);
        }
      }  
    } else {
      if (pred_size == 1) {
        for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
          const auto& sv = features.get_sparse_feature(ins_id);
          ret[ins_id - start_id] = this->model->predict_scalar(sv);
        }
      } else {
        for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
          const auto& sv = features.get_sparse_feature(ins_id);
          float* const preds = ret.data() + (ins_id - start_id) * pred_size;
          this->model->predict_vector(sv, preds);
        }
      }
    }
  }

  inline void LoadFromStream(std::istream& is) override {
    // TODO: override stream operators for GBDTModel
    std::string model_str;
    std::string temp;
    while (std::getline(is, temp)) {
      model_str += temp;
      model_str += '\n';
    }
    this->model.reset(new GBDTModel(*this->params));
    this->model->FromString(model_str);
  }

  inline void DumpToStream(std::ostream& os) override {
    // TODO: override stream operators for GBDTModel
    ASSERT(!this->is_empty()) << "Model is empty";
    os << this->model->ToString();
  }

  inline const char* name() const override { return "GBDT"; }

  inline int get_num_label() const override { return this->params->num_label; }
protected:
  inline bool is_empty() const {
    return this->model == nullptr;
  }

  std::unique_ptr<GBDTModel> model;
};

} // namespace gbdt
} // namespace ml
} // namespace hetu

# endif // __HETU_ML_MODEL_GBDT_GBDT_H_
