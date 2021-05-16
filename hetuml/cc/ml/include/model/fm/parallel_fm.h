#ifndef __HETU_ML_MODEL_FM_PARALLEL_FM_H_
#define __HETU_ML_MODEL_FM_PARALLEL_FM_H_

#include "model/fm/fm.h"
#include "ps/psmodel/PSVector.h"
#include "ps/psmodel/PSMatrix.h"

namespace hetu { 
namespace ml {
namespace fm {

template <typename Val>
class ParallelFM : public FM<Val> {
public:
  inline ParallelFM(const Args& args = {}): FM<Val>(args) {}

private:
  inline void InitModel(size_t max_dim) override {
    // sync max dim in order to avoid data skewness
    int rank = MyRank();
    int num_workers = NumWorkers();
    std::vector<Val> max_dims(num_workers);
    PSVector<Val> ps_max_dims("max_dims", num_workers);
    if (rank == 0) ps_max_dims.initAllZeros();
    PSAgent<Val>::Get()->barrier();
    auto local_max_dim = static_cast<Val>(max_dim);
    ps_max_dims.sparsePush(&rank, &local_max_dim, 1, false);
    PSAgent<Val>::Get()->barrier();
    ps_max_dims.densePull(max_dims.data(), num_workers);
    auto global_max_dim = static_cast<size_t>(*std::max_element(
      max_dims.begin(), max_dims.end()));

    // init local and ps model with global max dim
    FM<Val>::InitModel(global_max_dim);
    size_t embedding_dim = this->params->embedding_dim;
    this->ps_model.reset(new PSMatrix<Val>(
      "fm", embedding_dim + 1, global_max_dim));

    // local buffers
    this->bitmap.resize(global_max_dim);
    this->indices_buffer.resize(global_max_dim);
    this->compact_buffer.resize((embedding_dim + 1) * global_max_dim);
    this->values_buffer.resize(embedding_dim + 1);
    for (size_t vec_id = 0; vec_id <= embedding_dim; vec_id++) {
      // TODO: share the same memory with compact buffer
      this->values_buffer[vec_id].reset(new DenseVector<Val>(
        global_max_dim));
    }

    // leader worker initializes the model on PS
    if (rank == 0) {
      this->ps_model->initAllZeros();
      this->DensePushModel();
    }
    PSAgent<Val>::Get()->barrier();
  }

  inline Val FitOneEpoch(const Dataset<label_t, Val>& train_data) override {
    auto loss = FM<Val>::FitOneEpoch(train_data);
    // pull entire model from PS after each epoch
    this->DensePullModel();
    return loss;
  }

  Val FitOneBatch(const Dataset<label_t, Val>& train_data, 
                  size_t start_id, size_t end_id) override {
    // scan the non-zero entries
    auto num_nnz_dims = this->ScanNonzeros(train_data, start_id, end_id);
    // pull model from PS
    this->SparsePullModel(num_nnz_dims);
    
    // compute intermediate results
    std::vector<Val> inter;
    this->PredictInter(inter, train_data, start_id, end_id);
    // compute loss, grad, and update model
    Val batch_loss = 0;
    const auto* loss_func = LossFactory::GetLoss<Val>(this->params->loss);
    // accumulate gradients in buffer
    for (auto& vec_values_buffer : this->values_buffer) {
      vec_values_buffer->clear();
    }
    size_t batch_size = end_id - start_id;
    size_t embedding_dim = this->params->embedding_dim;
    auto step = -this->params->learning_rate / batch_size;
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      const auto& label = train_data.get_label(ins_id);
      const auto& feature = train_data.get_sparse_feature(ins_id);
      // combine into prediction
      auto offset = ins_id - start_id;
      // pred = <w, x> - 0.5 * sum_f(<v_f^2, x^2>) + 0.5 * sum(<v_f, x>^2)
      Val pred = inter[offset]; // <w, x> - 0.5 * sum_f(<v_f^2, x^2>)
      for (size_t embed_id = 0; embed_id < embedding_dim; embed_id++) {
        // <v_f, x>
        auto embed_dot = inter[(embed_id + 1) * batch_size + offset];
        pred += 0.5 * SQUARE(embed_dot);
      }
      // compute loss
      batch_loss += loss_func->loss(pred, label);
      Val grad_multipler = loss_func->grad(pred, label);
      // update of w
      this->values_buffer[0]->axp0(feature, step * grad_multipler);
      // update of v_f
      for (size_t embed_id = 0; embed_id < embedding_dim; embed_id++) {
        auto& embed = this->model[embed_id + 1];
        auto& vec_values_buffer = this->values_buffer[embed_id + 1];
        auto embed_dot = inter[(embed_id + 1) * batch_size + offset];
        for (size_t i = 0; i < feature.nnz; i++) {
          auto indice = feature.indices[i];
          auto value = feature.values[i];
          vec_values_buffer->values[indice] += \
            step * grad_multipler * value * \
            (embed_dot - embed->values[indice] * value);
        }
      }
    }
    
    // update model on PS
    this->SparsePushModel(num_nnz_dims);
    PSAgent<Val>::Get()->barrier();
    return batch_loss / (end_id - start_id);
  }

  inline void DensePullModel() {
    size_t num_rows = this->params->embedding_dim + 1;
    size_t num_cols = this->get_max_dim();
    // set all colIds
    std::iota(this->indices_buffer.begin(), this->indices_buffer.end(), 0);
    // pull from PS
    this->ps_model->pullCols(this->indices_buffer.data(), 
      this->compact_buffer.data(), num_cols, true);
    // copy to local model
    for (size_t vec_id = 0; vec_id < num_rows; vec_id++) {
      for (size_t dim = 0; dim < num_cols; dim++) {
        this->model[vec_id]->values[dim] = \
          this->compact_buffer[dim * num_rows + vec_id];
      }
    }
  }

  inline void DensePushModel() {
    size_t num_rows = this->params->embedding_dim + 1;
    size_t num_cols = this->get_max_dim();
    // set all colIds
    std::iota(this->indices_buffer.begin(), this->indices_buffer.end(), 0);
    // flatten the model by columns
    for (size_t dim = 0; dim < num_cols; dim++) {
      for (size_t vec_id = 0; vec_id < num_rows; vec_id++) {
        this->compact_buffer[dim * num_rows + vec_id] = \
          this->model[vec_id]->values[dim];
      }
    }
    // push to PS
    this->ps_model->pushCols(this->indices_buffer.data(), 
      this->compact_buffer.data(), num_cols, true);
  }

  inline void SparsePullModel(size_t num_pull_cols) {
    size_t num_rows = this->params->embedding_dim + 1;
    size_t num_cols = this->get_max_dim();
    // sparse pull from PS
    this->ps_model->pullCols(this->indices_buffer.data(), 
      this->compact_buffer.data(), num_pull_cols, false);
    // copy to local model
    for (size_t vec_id = 0; vec_id < num_rows; vec_id++) {
      for (size_t i = 0; i < num_pull_cols; i++) {
        size_t dim = this->indices_buffer[i];
        this->model[vec_id]->values[dim] = \
          this->compact_buffer[i * num_rows + vec_id];
      }
    }
  }

  inline void SparsePushModel(size_t num_push_cols) {
    size_t num_rows = this->params->embedding_dim + 1;
    size_t num_cols = this->get_max_dim();
    // flatten the local updates by columns
    for (size_t i = 0; i < num_push_cols; i++) {
      size_t dim = this->indices_buffer[i];
      for (size_t vec_id = 0; vec_id < num_rows; vec_id++) {
        this->compact_buffer[i * num_rows + vec_id] = \
          this->values_buffer[vec_id]->values[dim];
      }
    }
    // sparse push to PS
    this->ps_model->pushCols(this->indices_buffer.data(), 
      this->compact_buffer.data(), num_push_cols, true);
  }

  inline size_t ScanNonzeros(const Dataset<label_t, Val>& train_data, 
                             size_t start_id, size_t end_id) {
    // check the appearance of each dimension
    std::fill(this->bitmap.begin(), this->bitmap.end(), false);
    for (size_t ins_id = start_id; ins_id < end_id; ins_id++) {
      const auto& feature = train_data.get_sparse_feature(ins_id);
      for (size_t i = 0; i < feature.nnz; i++) {
        this->bitmap[feature.indices[i]] = true;
      }
    }
    // set to buffer
    size_t num_nnz_dims = 0;
    for (size_t dim = 0; dim < this->bitmap.size(); dim++) {
      if (this->bitmap[dim]) 
        this->indices_buffer[num_nnz_dims++] = dim;
    }
    return num_nnz_dims;
  }

  std::unique_ptr<PSMatrix<Val>> ps_model;
  
  // some workspace during fitting
  std::vector<bool> bitmap;
  std::vector<int> indices_buffer;
  std::vector<Val> compact_buffer;
  std::vector<std::unique_ptr<DenseVector<Val>>> values_buffer;
};

} // namespace fm
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_FM_PARALLEL_FM_H_
