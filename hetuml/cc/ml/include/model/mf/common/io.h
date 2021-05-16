#ifndef __HETU_ML_MODEL_MF_COMMON_IO_H_
#define __HETU_ML_MODEL_MF_COMMON_IO_H_

#include "common/logging.h"
#include "model/mf/common/common.h"
#include "model/mf/common/util.h"

namespace hetu { 
namespace ml {
namespace mf {

mf_problem read_problem(const std::string& path) {
  mf_problem prob;
  prob.m = 0;
  prob.n = 0;
  prob.nnz = 0;
  prob.R = nullptr;

  if (path.empty())
    return prob;

  std::ifstream f(path);
  ASSERT(f) << "Failed to open " << path;

  std::string line;
  while (getline(f, line))
    prob.nnz += 1;

  mf_node *R = new mf_node[static_cast<size_t>(prob.nnz)];

  f.close();
  
  f.open(path);

  long idx = 0;
  for (mf_node N; f >> N.u >> N.v >> N.r;) {
    if (N.u + 1 > prob.m)
      prob.m = N.u + 1;
    if (N.v + 1 > prob.n)
      prob.n = N.v + 1;
    R[idx] = N;
    ++idx;
  }
  prob.R = R;

  f.close();

  return prob;
}

void mf_save_model(const mf_model& model, std::ostream& os) {
  os << "f " << model.fun << std::endl;
  os << "m " << model.m << std::endl;
  os << "n " << model.n << std::endl;
  os << "k " << model.k << std::endl;
  os << "b " << model.b << std::endl;

  auto write = [&] (float *ptr, int size, char prefix) {
    for (int i = 0; i < size; ++i) {
      float *ptr1 = ptr + (long) i * model.k;
      os << prefix << i << " ";
      if (std::isnan(ptr1[0])) {
        os << "F ";
        for(int d = 0; d < model.k; ++d)
          os << 0 << " ";
      } else {
        os << "T ";
        for (int d = 0; d < model.k; ++d)
          os << ptr1[d] << " ";
      }
      os << std::endl;
    }
  };

  write(model.P, model.m, 'p');
  write(model.Q, model.n, 'q');
}

mf_model* mf_load_model(std::istream& is) {
  std::string dummy;

  mf_model *model = new mf_model;
  model->P = nullptr;
  model->Q = nullptr;

  is >> dummy >> model->fun 
    >> dummy >> model->m 
    >> dummy >> model->n 
    >> dummy >> model->k 
    >> dummy >> model->b;

  model->P = Utility::malloc_aligned_float((long) model->m * model->k);
  model->Q = Utility::malloc_aligned_float((long) model->n * model->k);

  auto read = [&] (float *ptr, int size) {
    for (int i = 0; i < size; ++i) {
      float *ptr1 = ptr + (long) i * model->k;
      is >> dummy >> dummy;
      if (dummy.compare("F") == 0) { // nan vector starts with "F"
        for (int d = 0; d < model->k; ++d) {
          is >> dummy;
          ptr1[d] = std::numeric_limits<float>::quiet_NaN();
        }
      } else {
        for (int d = 0; d < model->k; ++d)
          is >> ptr1[d];
      }
    }
  };

  read(model->P, model->m);
  read(model->Q, model->n);

  return model;
}

void mf_destroy_model(mf_model& model) {
  Utility::free_aligned_float(model.P);
  Utility::free_aligned_float(model.Q);
}

bool check_parameter(mf_parameter param) {
  if (param.fun != P_L2_MFR &&
      param.fun != P_L1_MFR &&
      param.fun != P_KL_MFR &&
      param.fun != P_LR_MFC &&
      param.fun != P_L2_MFC &&
      param.fun != P_L1_MFC &&
      param.fun != P_ROW_BPR_MFOC &&
      param.fun != P_COL_BPR_MFOC &&
      param.fun != P_L2_MFOC) {
    HML_LOG_FATAL << "unknown loss function";
    return false;
  }

  if (param.k < 1) {
    HML_LOG_FATAL << "number of factors must be greater than zero";
    return false;
  }

  if (param.nr_threads < 1) {
    HML_LOG_FATAL << "number of threads must be greater than zero";
    return false;
  }

  if (param.nr_bins < 1 || param.nr_bins < param.nr_threads) {
    HML_LOG_FATAL << "number of bins must be greater than number of threads";
    return false;
  }

  if (param.nr_iters < 1) {
    HML_LOG_FATAL << "number of iterations must be greater than zero";
    return false;
  }

  if (param.lambda_p1 < 0 ||
      param.lambda_p2 < 0 ||
      param.lambda_q1 < 0 ||
      param.lambda_q2 < 0) {
    HML_LOG_FATAL << "regularization coefficient must be non-negative";
    return false;
  }

  if (param.eta <= 0) {
    HML_LOG_FATAL << "learning rate must be greater than zero";
    return false;
  }

  if (param.fun == P_KL_MFR && !param.do_nmf) {
    HML_LOG_FATAL << "--nmf must be set when using generalized KL-divergence";
    return false;
  }

  if (param.nr_bins <= 2 * param.nr_threads) {
    HML_LOG_WARN << "insufficient blocks may slow down the training "
      << "process (4*nr_threads^2+1 blocks is suggested)";
  }

  if (param.alpha < 0) {
    HML_LOG_FATAL << "alpha must be a non-negative number";
    return false;
  }

  return true;
}



} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_COMMON_IO_H_
