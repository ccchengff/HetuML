#ifndef __HETU_ML_PYTHON_DATA_UTIL_H_
#define __HETU_ML_PYTHON_DATA_UTIL_H_

#include <pybind11/numpy.h>
#include <tuple>
#include "data/dataset.h"
#include "model/mf/common/io.h"
#include "model/lda/common/corpus.h"
#include "model/common/mlbase.h"

namespace py = pybind11;
using namespace hetu::ml;

template <typename IN, typename OUT>
inline py::array_t<OUT> ToPyArray(std::vector<IN>& vec) {
  auto result = py::array_t<OUT>(vec.size());
  py::buffer_info res_buff = result.request();
  auto* ptr = reinterpret_cast<OUT*>(res_buff.ptr);
  std::copy(vec.begin(), vec.end(), ptr);
  return result;
}

template <typename IN, typename OUT>
inline std::vector<OUT> FromPyArray(py::array_t<IN> arr) {
  py::buffer_info arr_buf = arr.request();
  ASSERT(arr_buf.ndim == 1) 
    << "Currently we can only transform one-dimensional arrays";
  auto arr_ptr = reinterpret_cast<IN*>(arr_buf.ptr);
  std::vector<OUT> result(arr_ptr, arr_ptr + arr_buf.shape[0]);
  return std::move(result);
}

template <typename IN, typename OUT>
inline std::vector<AVector<OUT>*> 
ParsePyCSRMatrix(py::array_t<int> indices, 
                 py::array_t<int> indptr, 
                 py::array_t<IN> values) {
  py::buffer_info indices_buf = indices.request();
  py::buffer_info indptr_buf = indptr.request();
  py::buffer_info values_buf = values.request();

  ASSERT(indices_buf.ndim == 1)
    << "Invalid number of dimension for indices: " << indices_buf.ndim;
  ASSERT(indptr_buf.ndim == 1)
    << "Invalid number of dimension for indptr: " << indptr_buf.ndim;
  ASSERT(values_buf.ndim == 1)
    << "Invalid number of dimension for values: " << values_buf.ndim;

  size_t num_ins = indptr_buf.shape[0] - 1;
  ASSERT(indptr_buf.shape[0] == num_ins + 1)
    << "Invalid shape: " << indptr_buf.shape[0] 
    << ", expected: " << num_ins + 1;
  ASSERT(indices_buf.shape[0] == values_buf.shape[0])
    << "Invalid shape: " << indices_buf.shape[0] 
    << ", expected: " << values_buf.shape[0];
  
  auto indices_ptr = reinterpret_cast<int*>(indices_buf.ptr);
  auto indptr_ptr = reinterpret_cast<int*>(indptr_buf.ptr);
  auto values_ptr = reinterpret_cast<IN*>(values_buf.ptr);

  std::vector<AVector<OUT>*> matrix;
  matrix.resize(num_ins, nullptr);
  for (size_t i = 0; i < num_ins; i++) {
    int offset = indptr_ptr[i];
    int nnz = indptr_ptr[i + 1] - offset;
    int* ins_indices = new int[nnz];
    OUT* ins_values = new OUT[nnz];
    for (int j = 0; j < nnz; j++) {
      // scipy automatically minus 1 here, we recover it for compatibility
      ins_indices[j] = indices_ptr[offset + j] + 1;
      ins_values[j] = values_ptr[offset + j];
    }
    int max_dim = nnz > 0 ? (ins_indices[nnz - 1] + 1) : 1;
    matrix[i] = new SparseVector<OUT>(ins_indices, ins_values, nnz, max_dim);
  }
  return matrix;
}

template <typename IN, typename OUT>
inline std::tuple<std::vector<int>, std::vector<int>, std::vector<OUT>> 
ParsePyCOOMatrix(py::array_t<int> rows, 
                 py::array_t<int> cols, 
                 py::array_t<IN> values) {
  py::buffer_info rows_buf = rows.request();
  py::buffer_info cols_buf = cols.request();
  py::buffer_info values_buf = values.request();

  ASSERT(rows_buf.ndim == 1)
    << "Invalid number of dimension for rows: " << rows_buf.ndim;
  ASSERT(cols_buf.ndim == 1)
    << "Invalid number of dimension for cols: " << cols_buf.ndim;
  ASSERT(values_buf.ndim == 1)
    << "Invalid number of dimension for values: " << values_buf.ndim;

  ASSERT(rows_buf.shape[0] == values_buf.shape[0])
    << "Invalid shape: " << rows_buf.shape[0] 
    << ", expected: " << values_buf.shape[0];
  ASSERT(cols_buf.shape[0] == values_buf.shape[0])
    << "Invalid shape: " << cols_buf.shape[0] 
    << ", expected: " << values_buf.shape[0];  
  size_t num_nnz = values_buf.shape[0];

  auto rows_ptr = reinterpret_cast<int*>(rows_buf.ptr);
  auto cols_ptr = reinterpret_cast<int*>(cols_buf.ptr);
  auto values_ptr = reinterpret_cast<IN*>(values_buf.ptr);

  std::vector<int> rows_vec(rows_ptr, rows_ptr + num_nnz);
  std::vector<int> cols_vec(cols_ptr, cols_ptr + num_nnz);
  std::vector<float> values_vec(values_ptr, values_ptr + num_nnz);
  return std::tuple<std::vector<int>, std::vector<int>, std::vector<OUT>>(
    rows_vec, cols_vec, values_vec);
}

template <typename IN, typename OUT>
inline DataMatrix<OUT>*
ParsePyDataMatrix(py::array_t<int> indices, 
                  py::array_t<int> indptr, 
                  py::array_t<IN> values) {
  auto features_vec = ParsePyCSRMatrix<IN, OUT>(indices, indptr, values);
  return new DataMatrix<OUT>(features_vec);
}

template <typename IN, typename OUT>
inline Dataset<label_t, OUT>* 
ParsePyDataset(py::array_t<label_t> labels, py::array_t<int> indices, 
               py::array_t<int> indptr, py::array_t<IN> values) {
  // parse labels
  auto labels_vec = FromPyArray<IN, label_t>(labels);
  // parse features
  auto features_vec = ParsePyCSRMatrix<IN, OUT>(indices, indptr, values);
  // return dataset
  return new Dataset<label_t, OUT>(labels_vec, features_vec);
}

template <typename V>
class DataMatrixWrapper {
public:
  DataMatrixWrapper(): matrix(nullptr) {}
  
  DataMatrixWrapper(py::array_t<int> indices, py::array_t<int> indptr, 
                    py::array_t<float> values) {
    this->matrix.reset(ParsePyDataMatrix<float, V>(indices, indptr, values));
  }
  
  inline uint32_t get_num_instances() const {
    return matrix == nullptr ? 0 : matrix->get_num_instances();
  }
  
  inline uint32_t get_max_dim() const {
    return matrix == nullptr ? 0 : matrix->get_max_dim();
  }

  inline bool is_dense() const {
    return matrix == nullptr ? false : matrix->is_dense();
  }
  
  std::unique_ptr<DataMatrix<V>> matrix;
};

template <typename V>
class DatasetWrapper {
public:
  DatasetWrapper(): dataset(nullptr) {}
  
  DatasetWrapper(const std::string& path, const std::string& data_format, 
                 bool neg_y, size_t rank, size_t num_workers) {
    this->dataset.reset(Dataset<label_t, V>::LoadData(
      path, data_format, neg_y, rank, num_workers));
  }

  DatasetWrapper(py::array_t<float> labels, py::array_t<int> indices, 
                 py::array_t<int> indptr, py::array_t<float> values) {
    this->dataset.reset(ParsePyDataset<float, V>(
      labels, indices, indptr, values));
  }
  
  inline size_t get_num_instances() const {
    return dataset == nullptr ? 0 : dataset->get_num_instances();
  }
  
  inline size_t get_max_dim() const {
    return dataset == nullptr ? 0 : dataset->get_max_dim();
  }

  inline bool is_dense() const {
    return dataset == nullptr ? false : dataset->is_dense();
  }
  
  std::unique_ptr<Dataset<label_t, V>> dataset;
};

template <typename V>
class COOMatrixWrapper {
public:
  COOMatrixWrapper(const std::string& path) {
    hetu::ml::mf::mf_problem prob = hetu::ml::mf::read_problem(path);
    this->matrix = std::make_unique<hetu::ml::mf::mf_problem>(prob);
    this->num_rows = this->matrix->m;
    this->num_cols = this->matrix->n;
    size_t nnz = this->matrix->nnz;
    this->rows.resize(nnz);
    this->cols.resize(nnz);
    this->values.resize(nnz);
    auto* nodes = this->matrix->R;
    for (size_t i = 0; i < nnz; i++) {
      this->rows[i] = nodes->u;
      this->cols[i] = nodes->v;
      this->values[i] = nodes->r;
    }
  }

  COOMatrixWrapper(py::array_t<int> rows, py::array_t<int> cols, 
                   py::array_t<float> values) {
    auto coo = ParsePyCOOMatrix<float, V>(rows, cols, values);
    this->rows = std::move(std::get<0>(coo));
    this->num_rows = *std::max_element(
      this->rows.begin(), this->rows.end()) + 1;
    this->cols = std::move(std::get<1>(coo));
    this->num_cols = *std::max_element(
      this->cols.begin(), this->cols.end()) + 1;
    this->values = std::move(std::get<2>(coo));
    this->matrix.reset(new hetu::ml::mf::mf_problem);
    size_t nnz = this->values.size();
    this->matrix->nnz = nnz;
    this->matrix->m = this->num_rows;
    this->matrix->n = this->num_cols;
    this->matrix->R = new hetu::ml::mf::mf_node[nnz];
    for (size_t i = 0; i < nnz; i++) {
      this->matrix->R[i].u = this->rows[i];
      this->matrix->R[i].v = this->cols[i];
      this->matrix->R[i].r = this->values[i];
    }
  }

  ~COOMatrixWrapper() {
    if (this->matrix != nullptr)
      delete[]this->matrix->R;
  }

  inline size_t get_num_rows() const {
    return num_rows;
  }

  inline size_t get_num_cols() const {
    return num_cols;
  }

  inline size_t get_num_kv() const {
    return values.size();
  }

  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<V> values;
  size_t num_rows;
  size_t num_cols;
  std::unique_ptr<hetu::ml::mf::mf_problem> matrix;
};

class CorpusWrapper {
public:
  CorpusWrapper(int n_words, const std::string& doc_path, 
                size_t rank, size_t num_workers) {
    this->corpus.reset(new hetu::ml::lda::Corpus(n_words));
    this->corpus->ReadFromFile(doc_path, rank, num_workers);
  }

  inline int get_num_docs() const {
    return corpus == nullptr ? 0 : corpus->n_docs;
  }

  inline int get_num_words() const {
    return corpus == nullptr ? 0 : corpus->n_words;
  }

  inline int get_num_tokens() const {
    return corpus == nullptr ? 0 : corpus->n_tokens;
  }

  inline int get_word_size(int id) const {
    return corpus == nullptr ? -1 : corpus->GetWordSize(id);
  }

  inline int get_doc_size(int id) const {
    return corpus == nullptr ? -1 : corpus->GetDocSize(id);
  }

  std::unique_ptr<hetu::ml::lda::Corpus> corpus;
};

#endif // __HETU_ML_PYTHON_DATA_UTIL_H_
