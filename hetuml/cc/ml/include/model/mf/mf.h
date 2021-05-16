#ifndef __HETU_ML_MODEL_MF_MF_H_
#define __HETU_ML_MODEL_MF_MF_H_

#include "model/common/mlbase.h"
#include "model/mf/common/api.h"

namespace hetu { 
namespace ml {
namespace mf {

namespace MFConf {
  // embedding dim
  static const std::string EMBEDDING_DIM = "EMBEDDING_DIM";
  static const int DEFAULT_EMBEDDING_DIM = 8;
  // number of epochs
  static const std::string NUM_EPOCH = "NUM_EPOCH";
  static const int DEFAULT_NUM_EPOCH = 10;
  // learning rate
  static const std::string LEARNING_RATE = "LEARNING_RATE";
  static const float DEFAULT_LEARNING_RATE = 0.1f;
  // L1 regularization
  static const std::string L1_REG = "L1_REG";
  static const float DEFAULT_L1_REG = 0.0f;
  // L2 regularization
  static const std::string L2_REG = "L2_REG";
  static const float DEFAULT_L2_REG = 0.0f; 

  static std::vector<std::string> meaningful_keys() {
    return {
      NUM_EPOCH, 
      EMBEDDING_DIM, 
      LEARNING_RATE, 
      L1_REG, 
      L2_REG
    };
  }

  static Args default_args() { 
    return {
      { NUM_EPOCH, std::to_string(DEFAULT_NUM_EPOCH) }, 
      { EMBEDDING_DIM, std::to_string(DEFAULT_EMBEDDING_DIM) }, 
      { LEARNING_RATE, std::to_string(DEFAULT_LEARNING_RATE) }, 
      { L1_REG, std::to_string(DEFAULT_L1_REG) }, 
      { L2_REG, std::to_string(DEFAULT_L2_REG) }
    };
  }
} // MFConf

class MFParam : public MLParam {
public:
  MFParam(const Args& args = {}, 
          const Args& default_args = MFConf::default_args())
  : MLParam(args, default_args) {
    this->InitAndCheckParam();
  }

  const std::vector<std::string> keys() const override { 
    return MFConf::meaningful_keys();
  }

  int num_epoch;
  int embedding_dim;
  float learning_rate;
  float l1reg;
  float l2reg;
protected:
  inline void InitAndCheckParam() override {
    this->num_epoch = argparse::Get<int>(
      this->all_args, MFConf::NUM_EPOCH);
    this->embedding_dim = argparse::Get<int>(
      this->all_args, MFConf::EMBEDDING_DIM);
    this->learning_rate = argparse::Get<float>(
      this->all_args, MFConf::LEARNING_RATE);
    this->l1reg = argparse::Get<float>(
      this->all_args, MFConf::L1_REG);
    this->l2reg = argparse::Get<float>(
      this->all_args, MFConf::L2_REG);
  }
};

class MF : public MLBase<MFParam> {
public:
  inline MF(const Args& args = {}): MLBase<MFParam>(args) {
    this->mf_param.reset(new mf_parameter);
    this->mf_param->k = this->params->embedding_dim;
    this->mf_param->nr_iters = this->params->num_epoch;
    this->mf_param->eta = this->params->learning_rate;
    this->mf_param->lambda_p1 = this->params->l1reg;
    this->mf_param->lambda_q1 = this->params->l1reg;
    this->mf_param->lambda_p2 = this->params->l2reg;
    this->mf_param->lambda_q2 = this->params->l2reg;
    this->mf_param->nr_workers = 1;
    // TODO: make these values configurable
    this->mf_param->fun = 0;
    this->mf_param->alpha = 1;
    this->mf_param->c = 0.0001;
    this->mf_param->nr_bins = 16;
    this->mf_param->nr_threads = 12;
    this->mf_param->quiet = false;
    this->mf_param->do_nmf = true;
  }

  inline ~MF() {
    if (this->model != nullptr) {
      mf_destroy_model(*this->model);
      this->model = nullptr;
    }
  }

  inline void Fit(const std::string& train_path, 
                  const std::string& valid_path = "") {
    mf_problem tr = read_problem(train_path);
    if (valid_path.length() > 0) {
      mf_problem va = read_problem(valid_path);
      this->Fit(&tr, &va);
      delete[]va.R;
    } else {
      this->Fit(&tr);
    }
    delete[]tr.R;
  }

  virtual inline void Fit(mf_problem const* train_data, 
                          mf_problem const* valid_data = nullptr) {
    TIK(fit);
    HML_LOG_INFO << "Start to fit " << this->name() << " model"
      << " with hyper-parameters:\n" << *this->params;
    if (valid_data != nullptr) {
      this->model.reset(mf_train_with_validation(
        train_data, valid_data, *mf_param));
    } else {
      this->model.reset(mf_train(train_data, *mf_param));
    }
    TOK(fit);
    HML_LOG_INFO << "Fit " << this->name() << " model"
      << " cost " << COST_MSEC(fit) << " ms";
  }

  inline void Predict(std::vector<float>& ret, const std::string& pred_path) {
    mf_problem pr = read_problem(pred_path);
    this->Predict(ret, pr, 0, pr.nnz);
    delete[]pr.R;
  }

  inline void Predict(std::vector<float>& ret, mf_problem& pred_data, 
                      size_t start_id, size_t end_id) {
    ret.resize(end_id - start_id);
    for (size_t i = start_id; i < end_id; i++) {
      ret[i - start_id] = mf_predict(*this->model, 
        pred_data.R[i].u, pred_data.R[i].v);
    }
  }

  inline const char* name() const override { return "MatrixFactorization"; }

protected:

  inline void LoadFromStream(std::istream& is) override {
    if (this->model != nullptr) {
      mf_destroy_model(*this->model);
      this->model = nullptr;
    }
    this->model.reset(mf_load_model(is));
  }

  inline void DumpToStream(std::ostream& os) override {
    ASSERT(!this->is_empty()) << "Model is empty";
    mf_save_model(*this->model, os);
  }

  inline bool is_empty() const { 
    return this->model == nullptr; 
  }

  std::unique_ptr<mf_parameter> mf_param = nullptr;
  std::unique_ptr<mf_model> model = nullptr;
};

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_MF_H_
