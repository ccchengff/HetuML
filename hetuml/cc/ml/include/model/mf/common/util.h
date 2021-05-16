#ifndef __HETU_ML_MODEL_MF_COMMON_UTIL_H_
#define __HETU_ML_MODEL_MF_COMMON_UTIL_H_

#include "model/mf/common/common.h"
#include "common/logging.h"
#include "common/threading.h"
#include <math.h>
#include <tuple>
#include <vector>
#include <random>
#include <stdexcept>
#include <limits>

namespace hetu { 
namespace ml {
namespace mf {

struct sort_node_by_p {
  bool operator() (mf_node const &lhs, mf_node const &rhs) {
    return std::tie(lhs.u, lhs.v) < std::tie(rhs.u, rhs.v);
  }
};

struct sort_node_by_q {
  bool operator() (mf_node const &lhs, mf_node const &rhs) {
    return std::tie(lhs.v, lhs.u) < std::tie(rhs.v, rhs.u);
  }
};

struct deleter {
  void operator() (mf_problem *prob) {
    delete[] prob->R;
    delete prob;
  }
};


class Utility {
public:
  Utility(int f, int n_threads) : fun(f), nr_threads(n_threads) {};
  
  void collect_info(mf_problem &prob, float &avg, float &std_dev) {
    double ex = 0;
    double ex2 = 0;

#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) \
  schedule(static) reduction(+:ex,ex2)
#endif
    for (long i = 0; i < prob.nnz; ++i) {
      mf_node &N = prob.R[i];
      ex += (double) N.r;
      ex2 += (double) N.r*N.r;
    }

    ex /= (double) prob.nnz;
    ex2 /= (double) prob.nnz;
    avg = (float) ex;
    std_dev = (float) std::sqrt(ex2-ex*ex);
  }
  
  
  void shuffle_problem(mf_problem &prob, std::vector<int> &p_map,
                       std::vector<int> &q_map) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(static)
#endif
    for (long i = 0; i < prob.nnz; i++) {
      mf_node &N = prob.R[i];
      if (N.u < (long) p_map.size())
        N.u = p_map[N.u];
      if (N.v < (long) q_map.size())
        N.v = q_map[N.v];
    }
  }
  
  std::vector<mf_node*> grid_problem(mf_problem &prob, int nr_bins,
                                     std::vector<int> &omega_p,
                                     std::vector<int> &omega_q,
                                     std::vector<Block> &blocks) {
    std::vector<long> counts(nr_bins * nr_bins, 0);

    int seg_p = (int) ceil((double) prob.m / nr_bins);
    int seg_q = (int) ceil((double) prob.n / nr_bins);

    auto get_block_id = [=] (int u, int v) {
      return (u / seg_p) * nr_bins + v / seg_q;
    };

    for (long i = 0; i < prob.nnz; ++i) {
      mf_node &N = prob.R[i];
      int block = get_block_id(N.u, N.v);
      counts[block] += 1;
      omega_p[N.u] += 1;
      omega_q[N.v] += 1;
    }

    std::vector<mf_node*> ptrs(nr_bins*nr_bins+1);
    mf_node *ptr = prob.R;
    ptrs[0] = ptr;
    for (int block = 0; block < nr_bins*nr_bins; ++block)
      ptrs[block+1] = ptrs[block] + counts[block];

    std::vector<mf_node*> pivots(ptrs.begin(), ptrs.end()-1);
    for (int block = 0; block < nr_bins*nr_bins; ++block) {
      for (mf_node* pivot = pivots[block]; pivot != ptrs[block+1];) {
        int curr_block = get_block_id(pivot->u, pivot->v);
        if (curr_block == block) {
          ++pivot;
          continue;
        }

        mf_node *next = pivots[curr_block];
        std::swap(*pivot, *next);
        pivots[curr_block] += 1;
      }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(dynamic)
#endif
    for (int block = 0; block < nr_bins*nr_bins; ++block) {
      if (prob.m > prob.n)
        std::sort(ptrs[block], ptrs[block+1], sort_node_by_p());
      else
        std::sort(ptrs[block], ptrs[block+1], sort_node_by_q());
    }

    for (int i = 0; i < (long) blocks.size(); ++i)
      blocks[i].tie_to(ptrs[i], ptrs[i+1]);

    return ptrs;
  }

  void scale_problem(mf_problem &prob, float scale) {
    if(scale == 1.0)
      return;

#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(static)
#endif
    for (long i = 0; i < prob.nnz; ++i)
      prob.R[i].r *= scale;
  }

  double calc_reg1(mf_model &model, float lambda_p, float lambda_q,
                   std::vector<int> &omega_p, std::vector<int> &omega_q) {
    auto calc_reg1_core = [&] (float *ptr, int size, std::vector<int> &omega) {
      double reg = 0;
      for (int i = 0; i < size; ++i) {
        if (omega[i] <= 0)
          continue;

        float tmp = 0;
        for (int j = 0; j < model.k; ++j)
          tmp += abs(ptr[(long)i * model.k + j]);
        reg += omega[i] * tmp;
      }
      return reg;
    };

    return lambda_p * calc_reg1_core(model.P, model.m, omega_p) + 
           lambda_q * calc_reg1_core(model.Q, model.n, omega_q);
  }
  
  double calc_reg2(mf_model &model, float lambda_p, float lambda_q,
                   std::vector<int> &omega_p, std::vector<int> &omega_q) {
    auto calc_reg2_core = [&] (float *ptr, int size, std::vector<int> &omega) {
      double reg = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:reg)
#endif
      for (int i = 0; i < size; ++i) {
        if (omega[i] <= 0)
          continue;

        float *ptr1 = ptr + (long) i * model.k;
        reg += omega[i] * Utility::inner_product(ptr1, ptr1, model.k);
      }

      return reg;
    };

    return lambda_p * calc_reg2_core(model.P, model.m, omega_p) +
           lambda_q * calc_reg2_core(model.Q, model.n, omega_q);
  }
  
  std::string get_error_legend() const {
    switch (fun) {
      case P_L2_MFR:
        return std::string("rmse");
        break;
      case P_L1_MFR:
        return std::string("mae");
        break;
      case P_KL_MFR:
        return std::string("gkl");
        break;
      case P_LR_MFC:
        return std::string("logloss");
        break;
      case P_L2_MFC:
      case P_L1_MFC:
        return std::string("accuracy");
        break;
      case P_ROW_BPR_MFOC:
      case P_COL_BPR_MFOC:
        return std::string("bprloss");
        break;
      case P_L2_MFOC:
        return std::string("sqerror");
      default:
        return std::string();
        break;
    }
  }
  
  double calc_error(std::vector<BlockBase*> &blocks,
                    std::vector<int> &cv_block_ids,
                    mf_model const &model) {
    double error = 0;
    if (fun == P_L2_MFR || fun == P_L1_MFR || fun == P_KL_MFR ||
        fun == P_LR_MFC || fun == P_L2_MFC || fun == P_L1_MFC) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) \
        reduction(+:error)
#endif
      for (int i = 0; i < (long) cv_block_ids.size(); ++i) {
        BlockBase *block = blocks[cv_block_ids[i]];
        block->reload();
        while (block->move_next()) {
          mf_node const &N = *(block->get_current());
          float z = mf_predict(model, N.u, N.v);
          switch (fun) {
            case P_L2_MFR:
              error += pow(N.r-z, 2);
              break;
            case P_L1_MFR:
              error += abs(N.r-z);
              break;
            case P_KL_MFR:
              error += N.r*log(N.r/z)-N.r+z;
              break;
            case P_LR_MFC:
              if (N.r > 0)
                  error += log(1.0 + exp(-z));
              else
                  error += log(1.0 + exp(z));
              break;
            case P_L2_MFC:
            case P_L1_MFC:
              if (N.r > 0)
                  error += z > 0 ? 1 : 0;
              else
                  error += z < 0 ? 1 : 0;
              break;
            default:
              throw std::invalid_argument("unknown error function");
              break;
          }
        }
        block->free();
      }
    } else {
      std::minstd_rand0 generator(rand());
      switch (fun) {
        case P_ROW_BPR_MFOC: {
          std::uniform_int_distribution<int> distribution(0, model.n-1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) \
        reduction(+:error)
#endif
          for (int i = 0; i < (long) cv_block_ids.size(); ++i) {
            BlockBase *block = blocks[cv_block_ids[i]];
            block->reload();
            while (block->move_next()) {
              mf_node const &N = *(block->get_current());
              int w = distribution(generator);
              error += log(1 + exp(mf_predict(model, N.u, w) - 
                mf_predict(model, N.u, N.v)));
            }
            block->free();
          }
          break;
        }
        case P_COL_BPR_MFOC: {
          std::uniform_int_distribution<int> distribution(0, model.m-1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) \
        reduction(+:error)
#endif
          for (int i = 0; i < (long) cv_block_ids.size(); ++i) {
            BlockBase *block = blocks[cv_block_ids[i]];
            block->reload();
            while (block->move_next()) {
              mf_node const &N = *(block->get_current());
              int w = distribution(generator);
              error += log(1 + exp(mf_predict(model, w, N.v) -
                mf_predict(model, N.u, N.v)));
            }
            block->free();
          }
          break;
        }
        default: {
          throw std::invalid_argument("unknown error function");
          break;
        }
      }
    }

    return error;
  }

  void scale_model(mf_model &model, float scale) {
    if(scale == 1.0)
      return;

    int k = model.k;

    model.b *= scale;

    auto scale1 = [&] (float *ptr, int size, float factor_scale) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nr_threads) schedule(static)
#endif
      for(int i = 0; i < size; ++i) {
        float *ptr1 = ptr + (long) i * model.k;
        for (int d = 0; d < k; ++d)
          ptr1[d] *= factor_scale;
      }
    };

    scale1(model.P, model.m, sqrt(scale));
    scale1(model.Q, model.n, sqrt(scale));
  }

  static mf_problem* copy_problem(mf_problem const *prob, bool copy_data) {
    mf_problem *new_prob = new mf_problem;

    if (prob == nullptr) {
      new_prob->m = 0;
      new_prob->n = 0;
      new_prob->nnz = 0;
      new_prob->R = nullptr;
    } else {
      new_prob->m = prob->m;
      new_prob->n = prob->n;
      new_prob->nnz = prob->nnz;
      if (copy_data) {
        try {
          new_prob->R = new mf_node[static_cast<size_t>(prob->nnz)];
          std::copy(prob->R, prob->R+prob->nnz, new_prob->R);
        } catch (...) {
          delete new_prob;
          throw;
        }
      } else {
        new_prob->R = prob->R;
      }
    }

    return new_prob;
  }
  
  static std::vector<int> gen_random_map(int size) {
    srand(0);
    std::vector<int> map(size, 0);
    for (int i = 0; i < size; ++i)
      map[i] = i;
    std::random_shuffle(map.begin(), map.end());
    return map;
  }
  
  // A function used to allocate all aligned float array.
  // It hides platform-specific function calls. Memory
  // allocated by malloc_aligned_float must be freed by using
  // free_aligned_float.
  static float* malloc_aligned_float(long size) {
    // Check if conversion from mf_long to size_t causes overflow.
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(float) + 1)
      throw std::bad_alloc();
    // [REVIEW] I hope one day we can use C11 aligned_alloc to replace
    // platform-depedent functions below. Both of Windows and OSX currently
    // don't support that function.
    void *ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(static_cast<size_t>(size*sizeof(float)), kALIGNByte);
#else
    int status = posix_memalign(&ptr, kALIGNByte, size * sizeof(float));
    if (status != 0)
      throw std::bad_alloc();
#endif
    if (ptr == nullptr)
      throw std::bad_alloc();

    return (float*) ptr;
  }
  
  // A function used to free all aligned float array.
  // It hides platform-specific function calls.
  static void free_aligned_float(float* ptr) {
#ifdef _WIN32
    // Unfortunately, Visual Studio doesn't want to support the
    // cross-platform allocation below.
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
  
  // Initialization function for stochastic gradient method.
  // Factor matrices P and Q are both randomly initialized.
  static mf_model* init_model(int fun, int m, int n, int k, float avg,
                              std::vector<int> &omega_p,
                              std::vector<int> &omega_q) {
    int k_real = k;
    int k_aligned = (int) ceil(double(k) / kALIGN) * kALIGN;

    mf_model *model = new mf_model;

    model->fun = fun;
    model->m = m;
    model->n = n;
    model->k = k_aligned;
    model->b = avg;
    model->P = nullptr;
    model->Q = nullptr;

    float scale = (float) sqrt(1.0 / k_real);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    try {
      model->P = Utility::malloc_aligned_float((long)model->m*model->k);
      model->Q = Utility::malloc_aligned_float((long)model->n*model->k);
    } catch(std::bad_alloc const &e) {
      HML_LOG_ERROR << e.what();
      Utility::free_aligned_float(model->P);
      Utility::free_aligned_float(model->Q);
      delete model;
      throw;
    }

    auto init1 = [&](float *start_ptr, long size, std::vector<int> counts) {
      memset(start_ptr, 0, static_cast<size_t>(
        sizeof(float) * size*model->k));
      for (long i = 0; i < size; ++i) {
        float * ptr = start_ptr + i * model->k;
        if (counts[static_cast<size_t>(i)] > 0) {
          for (long d = 0; d < k_real; ++d, ++ptr)
            *ptr = (float) (distribution(generator) * scale);
        } else if (fun != P_ROW_BPR_MFOC && fun != P_COL_BPR_MFOC) {
          // unseen for bpr is 0
          for (long d = 0; d < k_real; ++d, ++ptr)
            *ptr = std::numeric_limits<float>::quiet_NaN();
        }
      }
    };

    init1(model->P, m, omega_p);
    init1(model->Q, n, omega_q);

    return model;
}
  
  // Initialization function for one-class CD.
  // It does zero-initialization on factor matrix P and random initialization
  // on factor matrix Q.
  static mf_model* init_model(int m, int n, int k) {
    mf_model *model = new mf_model;

    model->fun = P_L2_MFOC;
    model->m = m;
    model->n = n;
    model->k = k;
    model->b = 0.0; // One-class matrix factorization doesn't have bias.
    model->P = nullptr;
    model->Q = nullptr;

    try {
      model->P = Utility::malloc_aligned_float((long)model->m * model->k);
      model->Q = Utility::malloc_aligned_float((long)model->n * model->k);
    } catch (std::bad_alloc const &e) {
      HML_LOG_ERROR << e.what();
      Utility::free_aligned_float(model->P);
      Utility::free_aligned_float(model->Q);
      delete model;
      throw;
    }

    // Our initialization strategy is that all P's elements are zero and do
    // random initization on Q. Thus, all initial predicted ratings are all zero
    // since the approximated rating matrix is PQ^T.

    // Initialize P with zeros
    for (long i = 0; i < k * m; ++i)
      model->P[i] = 0.0;

    // Initialize Q with random numbers
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (long i = 0; i < k * n; ++i)
      model->Q[i] = distribution(generator);

    return model;
  }
  
  static float inner_product(float *p, float *q, int k) {
    return std::inner_product(p, p+k, q, (float)0.0);
  }
  
  static std::vector<int> gen_inv_map(std::vector<int> &map) {
    std::vector<int> inv_map(map.size());
    for (int i = 0; i < (long) map.size(); ++i)
      inv_map[map[i]] = i;
    return inv_map;
  }
  
  static void shrink_model(mf_model &model, int k_new) {
    int k_old = model.k;
    model.k = k_new;

    auto shrink1 = [&] (float *ptr, int size) {
      for (int i = 0; i < size; ++i) {
        float *src = ptr + (long) i * k_old;
        float *dst = ptr + (long) i * k_new;
        std::copy(src, src + k_new, dst);
      }
    };

    shrink1(model.P, model.m);
    shrink1(model.Q, model.n);
  }
  
  static void shuffle_model(mf_model &model,
                            std::vector<int> &p_map,
                            std::vector<int> &q_map) {
    auto inv_shuffle1 = [] (float *vec, std::vector<int> &map,
                            int size, int k) {
      for (int pivot = 0; pivot < size;) {
        if (pivot == map[pivot]) {
          ++pivot;
          continue;
        }

        int next = map[pivot];

        for (int d = 0; d < k; ++d)
          std::swap(*(vec + (long) pivot * k + d), *(vec + (long) next * k + d));

        map[pivot] = map[next];
        map[next] = next;
      }
    };

    inv_shuffle1(model.P, p_map, model.m, model.k);
    inv_shuffle1(model.Q, q_map, model.n, model.k);
  }
  
  int get_thread_number() const { return nr_threads; };

private:
  int fun;
  int nr_threads;
};

class ModelAverageBase {
public:
  virtual void Init(mf_model& model) = 0;

  virtual void UpdateModel(mf_model& model) = 0;

  virtual void Finalize(mf_model& model) = 0;
};

static ModelAverageBase* modelAverge = nullptr;

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_COMMON_UTIL_H_
