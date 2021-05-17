#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "python_binding/data_util.h"

#include "model/linear/linear.h"
#include "model/fm/fm.h"
#include "model/naive_bayes/naive_bayes.h"
#include "model/gbdt/gbdt.h"
#include "model/rf/rf.h"
#include "model/knn/knn.h"
#include "model/mf/mf.h"
#include "model/lda/lda.h"

#ifdef WITH_PARALLEL
#include "model/linear/parallel_linear.h"
#include "model/fm/parallel_fm.h"
#include "model/naive_bayes/parallel_naive_bayes.h"
#include "model/gbdt/parallel_gbdt.h"
#include "model/rf/parallel_rf.h"
#include "model/mf/parallel_mf.h"
#include "model/lda/parallel_lda.h"
#include "ps/psf/PSFHandle.h"
#endif

using namespace hetu::ml;
using namespace hetu::ml::linear;
using namespace hetu::ml::fm;
using namespace hetu::ml::naive_bayes;
using namespace hetu::ml::gbdt;
using namespace hetu::ml::rf;
using namespace hetu::ml::knn;
using namespace hetu::ml::mf;
using namespace hetu::ml::lda;

typedef float value_t;

#define DISPATCH_HETU_ML_MODEL(__MODEL__, __NAME__, module) \
  py::class_<__MODEL__>(module, __NAME__)                   \
    .def(py::init())                                        \
    .def(py::init<const Args&>())                           \
    .def("GetParams", [](__MODEL__& self) {                 \
      return self.get_params().get_all_args();              \
    })                                                      \
    .def("SaveModel", &__MODEL__::SaveModel)                \
    .def("LoadModel", &__MODEL__::LoadModel)                \
    .def("name", &__MODEL__::name)

#define DISPATCH_HETU_SUPERVISED_ML_MODEL(__MODEL__, __NAME__, module)  \
  DISPATCH_HETU_ML_MODEL(__MODEL__, __NAME__, module)                   \
    .def("Fit0", [](__MODEL__& self,                                    \
                    const std::string& train_path,                      \
                    const std::string& valid_path,                      \
                    const std::string& data_type,                       \
                    size_t rank, size_t num_workers) {                  \
      auto* train_data = Dataset<label_t, value_t>::LoadData(           \
        train_path, data_type, self.use_neg_y());                       \
      if (valid_path.empty()) {                                         \
        self.Fit(*train_data);                                          \
      } else {                                                          \
        auto* valid_data = Dataset<label_t, value_t>::LoadData(         \
          valid_path, data_type, self.use_neg_y());                     \
        self.Fit(*train_data, *valid_data);                             \
        delete valid_data;                                              \
      }                                                                 \
      delete train_data;                                                \
    }, py::arg("train_path"), py::arg("valid_path") = py::str(),        \
       py::arg("data_type") = py::str("libsvm"),                        \
       py::arg("rank") = 0,                                    \
       py::arg("num_workers") = 1)                             \
    .def("Fit1", [](__MODEL__& self,                                    \
                    const DatasetWrapper<value_t>& train_data) {        \
      ASSERT(train_data.dataset != nullptr) << "Train data is empty";   \
      self.Fit(*train_data.dataset);                                    \
    }, py::arg("train_data"))                                           \
    .def("Fit2", [](__MODEL__& self,                                    \
                    const DatasetWrapper<value_t>& train_data,          \
                    const DatasetWrapper<value_t>& valid_data) {        \
      ASSERT(train_data.dataset != nullptr) << "Train data is empty";   \
      if (valid_data.dataset != nullptr)                                \
        self.Fit(*train_data.dataset, *valid_data.dataset);             \
      else                                                              \
        self.Fit(*train_data.dataset);                                  \
    }, py::arg("train_data"), py::arg("valid_data"))                    \
    .def("Predict0", [](__MODEL__& self, const std::string& pred_path) {\
      auto* pred_data = Dataset<label_t, value_t>::LoadData(            \
        pred_path, "libsvm", self.use_neg_y());                         \
      auto num_ins = pred_data->get_num_instances();                    \
      std::vector<label_t> preds;                                       \
      self.Predict(preds, *pred_data, 0, num_ins);                      \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      delete pred_data;                                                 \
      return ret;                                                       \
    }, py::arg("pred_path"))                                            \
    .def("Predict1", [](__MODEL__& self,                                \
                        const DataMatrixWrapper<value_t>& pred_data) {  \
      ASSERT(pred_data.matrix != nullptr) << "Pred data is empty";      \
      auto num_ins = pred_data.get_num_instances();                     \
      std::vector<label_t> preds;                                       \
      self.Predict(preds, *pred_data.matrix, 0, num_ins);               \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      return ret;                                                       \
    }, py::arg("pred_data"))                                            \
    .def("Predict2", [](__MODEL__& self,                                \
                        const DatasetWrapper<value_t>& pred_data) {     \
      ASSERT(pred_data.dataset != nullptr) << "Pred data is empty";     \
      auto num_ins = pred_data.get_num_instances();                     \
      std::vector<label_t> preds;                                       \
      self.Predict(preds, *pred_data.dataset, 0, num_ins);              \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      return ret;                                                       \
    }, py::arg("pred_data"))                                            \
    .def("Evaluate0", [](__MODEL__& self,                               \
                         const DatasetWrapper<value_t>& eval_data,      \
                         const std::vector<std::string>& metrics) {     \
      ASSERT(eval_data.dataset != nullptr) << "Eval data is empty";     \
      std::vector<value_t> m = self.Evaluate(                           \
        *eval_data.dataset, metrics);                                   \
      py::array_t<value_t> ret = ToPyArray<value_t, value_t>(m);        \
      return ret;                                                       \
    }, py::arg("eval_data"), py::arg("metrics"))

#define DISPATCH_HETU_KNN_MODEL(__MODEL__, __NAME__, module)            \
  DISPATCH_HETU_ML_MODEL(__MODEL__, __NAME__, module)                   \
    .def("Predict0", [](__MODEL__& self,                                \
                        const std::string& train_path,                  \
                        const std::string& pred_path) {                 \
      auto* train_data =                                                \
        Dataset<label_t, value_t>::LoadData(train_path, "libsvm");      \
      auto* pred_data =                                                 \
        Dataset<label_t, value_t>::LoadData(pred_path, "libsvm");       \
      std::vector<label_t> preds;                                       \
      self.Predict(preds, *train_data, *pred_data);                     \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      return ret;                                                       \
    }, py::arg("train_path"), py::arg("pred_path"))                     \
    .def("Predict1", [](__MODEL__& self,                                \
                        const DatasetWrapper<value_t>& train_data,      \
                        const DataMatrixWrapper<value_t>& pred_data) {  \
      std::vector<label_t> preds;                                       \
      self.Predict(preds, *train_data.dataset, *pred_data.matrix);      \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      return ret;                                                       \
    }, py::arg("train_data"), py::arg("pred_data"))                     \
    .def("Predict2", [](__MODEL__& self,                                \
                        const DatasetWrapper<value_t>& train_data,      \
                        const DatasetWrapper<value_t>& pred_data) {     \
      std::vector<label_t> preds;                                       \
      self.Predict(preds, *train_data.dataset, *pred_data.dataset);     \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      return ret;                                                       \
    }, py::arg("train_data"), py::arg("pred_data"))                     \
    .def("Evaluate0", [](__MODEL__& self,                               \
                        const DatasetWrapper<value_t>& train_data,      \
                        const DatasetWrapper<value_t>& eval_data,       \
                        const std::vector<std::string>& metrics) {      \
      std::vector<value_t> m = self.Evaluate(                           \
        *train_data.dataset, *eval_data.dataset, metrics);              \
      py::array_t<value_t> ret = ToPyArray<value_t, value_t>(m);        \
      return ret;                                                       \
    }, py::arg("train_data"), py::arg("eval_data"), py::arg("metrics"))

#define DISPATCH_HETU_MF_MODEL(__MODEL__, __NAME__, module)             \
  DISPATCH_HETU_ML_MODEL(__MODEL__, __NAME__, module)                   \
    .def("Fit0", [](__MODEL__& self,                                    \
                    const std::string& train_path,                      \
                    const std::string& valid_path) {                    \
      COOMatrixWrapper<value_t> train_data(train_path);                 \
      if (valid_path.length() > 0) {                                    \
        COOMatrixWrapper<value_t> valid_data(valid_path);               \
        self.Fit(train_data.matrix.get(), valid_data.matrix.get());     \
      } else {                                                          \
        self.Fit(train_data.matrix.get());                              \
      }                                                                 \
    }, py::arg("train_path"), py::arg("valid_path") = py::str())        \
    .def("Fit1", [](__MODEL__& self,                                    \
                    const COOMatrixWrapper<value_t>& train_data) {      \
      self.Fit(train_data.matrix.get());                                \
    }, py::arg("train_data"))                                           \
    .def("Fit2", [](__MODEL__& self,                                    \
                    const COOMatrixWrapper<value_t>& train_data,        \
                    const COOMatrixWrapper<value_t>& valid_data) {      \
      self.Fit(train_data.matrix.get(), valid_data.matrix.get());       \
    }, py::arg("train_data"), py::arg("valid_data"))                    \
    .def("Predict0", [](__MODEL__& self,                                \
                        const std::string& pred_path) {                 \
      std::vector<label_t> preds;                                       \
      self.Predict(preds, pred_path);                                   \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      return ret;                                                       \
    }, py::arg("pred_path"))                                            \
    .def("Predict1", [](__MODEL__& self,                                \
                        const COOMatrixWrapper<value_t>& pred_data) {   \
      std::vector<label_t> preds;                                       \
      size_t nnz = pred_data.get_num_kv();                              \
      self.Predict(preds, *pred_data.matrix, 0, nnz);                   \
      py::array_t<label_t> ret = ToPyArray<label_t, label_t>(preds);    \
      return ret;                                                       \
    }, py::arg("pred_data"))

#define DISPATCH_HETU_LDA_MODEL(__MODEL__, __NAME__, module)            \
  DISPATCH_HETU_ML_MODEL(__MODEL__, __NAME__, module)                   \
    .def("Fit0", [](__MODEL__& self,                                    \
                    const std::string& train_path) {                    \
      Corpus corpus(self.get_params().num_words);                       \
      corpus.ReadFromFile(train_path);                                  \
      self.Fit(corpus);                                                 \
    }, py::arg("train_path"))                                           \
    .def("Fit1", [](__MODEL__& self,                                    \
                    const CorpusWrapper& train_data) {                  \
      self.Fit(*train_data.corpus);                                     \
    }, py::arg("train_data"))                                           \
    .def("Predict0", [](__MODEL__& self,                                \
                        const std::string& pred_path) {                 \
      Corpus corpus(self.get_params().num_words);                       \
      corpus.ReadFromFile(pred_path);                                   \
      std::vector<int> preds = self.Predict(corpus);                    \
      py::array_t<int> ret = ToPyArray<int, int>(preds);                \
      return ret;                                                       \
    }, py::arg("pred_path"))                                            \
    .def("Predict1", [](__MODEL__& self,                                \
                        const CorpusWrapper& pred_data) {               \
      std::vector<int> preds = self.Predict(*pred_data.corpus);         \
      py::array_t<int> ret = ToPyArray<int, int>(preds);                \
      return ret;                                                       \
    }, py::arg("pred_data"))

#ifdef WITH_PARALLEL
template <typename HetuMLPyPsDataType>
void StartServer() {
  ASSERT(ps::IsServer()) 
    << "Only server role can start parameter servers";
  auto server = new KVServer<HetuMLPyPsDataType>(0);
  server->set_request_handle(KVServerMatrixHandle<HetuMLPyPsDataType>());
  RegisterExitCallback([server]() { delete server; });
}

#define DISPATCH_HETU_PS(module) \
  module \
    .def("InitPS", []() { ps::Start(0); })               \
    .def("FinalizePS", []() { ps::Finalize(0, true); })  \
    .def("NumWorkers", &ps::NumWorkers)                  \
    .def("NumServers", &ps::NumServers)                  \
    .def("IsWorker", &ps::IsWorker)                      \
    .def("IsServer", &ps::IsServer)                      \
    .def("IsScheduler", &ps::IsScheduler)                \
    .def("MyRank", &ps::MyRank)                          \
    .def("StartIntServer", &StartServer<int>)            \
    .def("StartFloatServer", &StartServer<float>)        \
    .def("StartDoubleServer", &StartServer<double>)      \
    .def("_with_parallel", []() { return true; })
#endif


PYBIND11_MODULE(hetuml_core, m) {
  /* data */
  py::class_<DataMatrixWrapper<value_t>>(m, "DataMatrixWrapper")
    .def(py::init<py::array_t<int>, py::array_t<int>, py::array_t<value_t>>())
    .def("get_num_instances", &DataMatrixWrapper<value_t>::get_num_instances)
    .def("get_max_dim", &DataMatrixWrapper<value_t>::get_max_dim)
    .def("is_dense", &DataMatrixWrapper<value_t>::is_dense);
  py::class_<DatasetWrapper<value_t>>(m, "DatasetWrapper")
    .def(py::init<const std::string&, const std::string&, 
                  bool, size_t, size_t>())
    .def(py::init<py::array_t<value_t>, py::array_t<int>, 
                  py::array_t<int>, py::array_t<value_t>>())
    .def("get_num_instances", &DatasetWrapper<value_t>::get_num_instances)
    .def("get_max_dim", &DatasetWrapper<value_t>::get_max_dim)
    .def("is_dense", &DatasetWrapper<value_t>::is_dense);
  py::class_<COOMatrixWrapper<value_t>>(m, "COOMatrixWrapper")
    .def(py::init<py::array_t<int>, py::array_t<int>, py::array_t<value_t>>())
    .def(py::init<const std::string&>())
    .def("get_num_rows", &COOMatrixWrapper<value_t>::get_num_rows)
    .def("get_num_cols", &COOMatrixWrapper<value_t>::get_num_rows)
    .def("get_num_kv", &COOMatrixWrapper<value_t>::get_num_rows);
  py::class_<CorpusWrapper>(m, "CorpusWrapper")
    .def(py::init<int, const std::string&, size_t, size_t>())
    .def("get_num_docs", &CorpusWrapper::get_num_docs)
    .def("get_num_words", &CorpusWrapper::get_num_words)
    .def("get_num_tokens", &CorpusWrapper::get_num_tokens)
    .def("get_word_size", &CorpusWrapper::get_word_size)
    .def("get_doc_size", &CorpusWrapper::get_doc_size);

  /* supervised ML algorithms */
  DISPATCH_HETU_SUPERVISED_ML_MODEL(LogReg<value_t>, "LR", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(SVM<value_t>, "SVM", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(LinearReg<value_t>, "LinearReg", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(FM<value_t>, "FM", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(NaiveBayes<value_t>, "NaiveBayes", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(GBDT<value_t>, "GBDT", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(RF<value_t>, "RF", m);
  /* kNN */
  DISPATCH_HETU_KNN_MODEL(KNN<value_t>, "KNN", m);
  /* MF */
  DISPATCH_HETU_MF_MODEL(MF, "MF", m);
  /* LDA */
  DISPATCH_HETU_LDA_MODEL(LDA, "LDA", m);

#ifdef WITH_PARALLEL
  DISPATCH_HETU_PS(m);
  /* parallel supervised ML algorithms */
  DISPATCH_HETU_SUPERVISED_ML_MODEL(ParallelLogReg<value_t>, "ParallelLR", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(ParallelSVM<value_t>, "ParallelSVM", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(ParallelLinearReg<value_t>, "ParallelLinearReg", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(ParallelFM<value_t>, "ParallelFM", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(ParallelNaiveBayes<value_t>, "ParallelNaiveBayes", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(ParallelGBDT<value_t>, "ParallelGBDT", m);
  DISPATCH_HETU_SUPERVISED_ML_MODEL(ParallelRF<value_t>, "ParallelRF", m);
  /* parallel MF */
  DISPATCH_HETU_MF_MODEL(ParallelMF, "ParallelMF", m);
  /* parallel LDA */
  DISPATCH_HETU_LDA_MODEL(ParallelLDA, "ParallelLDA", m);
#endif
}
