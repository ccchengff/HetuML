#ifndef __HETU_ML_MODEL_MF_COMMON_API_H_
#define __HETU_ML_MODEL_MF_COMMON_API_H_

#include "common/logging.h"
#include "model/mf/common/common.h"
#include "model/mf/common/solver.h"
#include "model/mf/common/util.h"
#include "model/mf/common/io.h"

namespace hetu { 
namespace ml {
namespace mf {

void fpsg_core(Utility &util, Scheduler &sched,
               mf_problem *tr, mf_problem *va, 
               mf_parameter param, float scale,
               std::vector<BlockBase*> &block_ptrs,
               std::vector<int> &omega_p,
               std::vector<int> &omega_q,
               std::shared_ptr<mf_model> &model,
               std::vector<int> cv_blocks,
               double *cv_error) {
  ASSERT_GT(tr->nnz, 0) << "No training data";

  if (param.fun == P_L2_MFR ||
      param.fun == P_L1_MFR ||
      param.fun == P_KL_MFR) {
    switch (param.fun) {
      case P_L2_MFR:
        param.lambda_p2 /= scale;
        param.lambda_q2 /= scale;
        param.lambda_p1 /= (float) pow(scale, 1.5);
        param.lambda_q1 /= (float) pow(scale, 1.5);
        break;
      case P_L1_MFR:
      case P_KL_MFR:
        param.lambda_p1 /= sqrt(scale);
        param.lambda_q1 /= sqrt(scale);
        break;
    }
  }

  if (modelAverge != nullptr) {
    modelAverge->Init(*model);
  }

  bool slow_only = (param.lambda_p1 == 0 && param.lambda_q1 == 0);
  std::vector<float> PG(model->m * 2, 1), QG(model->n * 2, 1);

  std::vector<std::shared_ptr<SolverBase>> solvers(param.nr_threads);
  std::vector<std::thread> threads;
  threads.reserve(param.nr_threads);
  for (int i = 0; i < param.nr_threads; ++i) {
    solvers[i] = SolverFactory::get_solver(sched, block_ptrs,
      PG.data(), QG.data(), *model, param, slow_only);
    threads.emplace_back(&SolverBase::run, solvers[i].get());
  }

  auto error_legend = util.get_error_legend();
  for (int iter = 0; iter < param.nr_iters; ++iter) {
    sched.wait_for_jobs_done();

    if (!param.quiet) {
      double reg = 0;
      double reg1 = util.calc_reg1(*model, param.lambda_p1, param.lambda_q1, 
        omega_p, omega_q);
      double reg2 = util.calc_reg2(*model, param.lambda_p2, param.lambda_q2, 
        omega_p, omega_q);
      double tr_loss = sched.get_loss();
      double tr_error = sched.get_error() / tr->nnz;

      switch (param.fun) {
        case P_L2_MFR:
          reg = (reg1+reg2)*scale*scale;
          tr_loss *= scale*scale;
          tr_error = sqrt(tr_error*scale*scale);
          break;
        case P_L1_MFR:
        case P_KL_MFR:
          reg = (reg1+reg2)*scale;
          tr_loss *= scale;
          tr_error *= scale;
          break;
        default:
          reg = reg1+reg2;
          break;
      }
      HML_LOG_INFO << "Epoch[" << iter << "] Train " 
        << "loss[" << (float) (reg + tr_loss) << "] "
        << error_legend << "[" << (float) tr_error << "]";

      if (va->nnz != 0) {
        Block va_block(va->R, va->R + va->nnz);
        std::vector<BlockBase*> va_blocks(1, &va_block);
        std::vector<int> va_block_ids(1, 0);
        double va_error = util.calc_error(va_blocks, va_block_ids, 
                                          *model) / va->nnz;
        switch (param.fun) {
          case P_L2_MFR:
            va_error = sqrt(va_error*scale*scale);
            break;
          case P_L1_MFR:
          case P_KL_MFR:
            va_error *= scale;
            break;
        }

        HML_LOG_INFO << "Epoch[" << iter << "] Valid " 
        << error_legend << "[" << (float) va_error << "]";
      }
    }

    if (modelAverge != nullptr) {
      modelAverge->UpdateModel(*model);
    }

    if (iter == 0)
      slow_only = false;
    if (iter == param.nr_iters - 1)
      sched.terminate();
    sched.resume();
  }

  if (modelAverge != nullptr) {
    modelAverge->Finalize(*model);
  }
  
  for (auto &thread : threads)
    thread.join();

  if (cv_error != nullptr && cv_blocks.size() > 0) {
    long cv_count = 0;
    for(auto block : cv_blocks)
      cv_count += block_ptrs[block]->get_nnz();

    *cv_error = util.calc_error(block_ptrs, cv_blocks, *model) / cv_count;

    switch(param.fun) {
      case P_L2_MFR:
        *cv_error = sqrt(*cv_error*scale*scale);
        break;
      case P_L1_MFR:
      case P_KL_MFR:
        *cv_error *= scale;
        break;
    }
  }
}

std::shared_ptr<mf_model> fpsg(mf_problem const *tr_, mf_problem const *va_,
                               mf_parameter param, 
                               std::vector<int> cv_blocks=std::vector<int>(),
                               double *cv_error=nullptr) {
  std::shared_ptr<mf_model> model;

  Utility util(param.fun, param.nr_threads);
  Scheduler sched(param.nr_bins, param.nr_threads, cv_blocks);
  std::shared_ptr<mf_problem> tr;
  std::shared_ptr<mf_problem> va;
  std::vector<Block> blocks(param.nr_bins * param.nr_bins);
  std::vector<BlockBase*> block_ptrs(param.nr_bins * param.nr_bins);
  std::vector<mf_node*> ptrs;
  std::vector<int> p_map;
  std::vector<int> q_map;
  std::vector<int> inv_p_map;
  std::vector<int> inv_q_map;
  std::vector<int> omega_p;
  std::vector<int> omega_q;
  float avg = 0;
  float std_dev = 0;
  float scale = 1;

  tr = std::shared_ptr<mf_problem>(Utility::copy_problem(tr_, false));
  va = std::shared_ptr<mf_problem>(Utility::copy_problem(va_, false));

  util.collect_info(*tr, avg, std_dev);

  if (param.fun == P_L2_MFR ||
      param.fun == P_L1_MFR ||
      param.fun == P_KL_MFR) {
    scale = std::max((float) 1e-4, std_dev);
  }

  p_map = Utility::gen_random_map(tr->m);
  q_map = Utility::gen_random_map(tr->n);
  inv_p_map = Utility::gen_inv_map(p_map);
  inv_q_map = Utility::gen_inv_map(q_map);
  omega_p = std::vector<int>(tr->m, 0);
  omega_q = std::vector<int>(tr->n, 0);

  util.shuffle_problem(*tr, p_map, q_map);
  util.shuffle_problem(*va, p_map, q_map);
  util.scale_problem(*tr, 1.0f / scale);
  util.scale_problem(*va, 1.0f / scale);
  ptrs = util.grid_problem(*tr, param.nr_bins, omega_p, omega_q, blocks);

  model = std::shared_ptr<mf_model>(Utility::init_model(param.fun,
              tr->m, tr->n, param.k, avg/scale, omega_p, omega_q),
              [] (mf_model *ptr) { mf_destroy_model(*ptr); });

  for (int i = 0; i < (long) blocks.size(); ++i)
    block_ptrs[i] = &blocks[i];

  fpsg_core(util, sched, tr.get(), va.get(), param, scale,
            block_ptrs, omega_p, omega_q, model, cv_blocks, cv_error);

  util.scale_problem(*tr, scale);
  util.scale_problem(*va, scale);
  util.shuffle_problem(*tr, inv_p_map, inv_q_map);
  util.shuffle_problem(*va, inv_p_map, inv_q_map);

  util.scale_model(*model, scale);
  Utility::shrink_model(*model, param.k);
  Utility::shuffle_model(*model, inv_p_map, inv_q_map);
  
  return model;
}


class CrossValidatorBase {
public:
  CrossValidatorBase(mf_parameter param_, int nr_folds_)
  : param(param_), nr_bins(param_.nr_bins), nr_folds(nr_folds_),
      nr_blocks_per_fold(nr_bins*nr_bins/nr_folds), quiet(param_.quiet),
      util(param.fun, param.nr_threads), cv_error(0) {
    param.quiet = true;
  }
  
  double do_cross_validation() {
    std::vector<int> cv_blocks;
    srand(0);
    for (int block = 0; block < nr_bins * nr_bins; ++block)
      cv_blocks.push_back(block);
    std::random_shuffle(cv_blocks.begin(), cv_blocks.end());

    if(!quiet) {
      std::cout.width(4);
      std::cout << "fold";
      std::cout.width(10);
      std::cout << util.get_error_legend();
      std::cout << std::endl;
    }

    cv_error = 0;

    for (int fold = 0; fold < nr_folds; ++fold) {
      int begin = fold * nr_blocks_per_fold;
      int end = std::min((fold+1)*nr_blocks_per_fold, nr_bins*nr_bins);
      std::vector<int> hidden_blocks(cv_blocks.begin() + begin, 
                                     cv_blocks.begin() + end);

      double err = do_cv1(hidden_blocks);
      cv_error += err;

      if (!quiet) {
        std::cout.width(4);
        std::cout << fold;
        std::cout.width(10);
        std::cout << std::fixed << std::setprecision(4) << err;
        std::cout << std::endl;
      }
    }

    if (!quiet) {
      std::cout.width(14);
      std::cout.fill('=');
      std::cout << "" << std::endl;
      std::cout.fill(' ');
      std::cout.width(4);
      std::cout << "avg";
      std::cout.width(10);
      std::cout << std::fixed << std::setprecision(4) << cv_error / nr_folds;
      std::cout << std::endl;
    }

    return cv_error / nr_folds;
  }
  
  virtual double do_cv1(std::vector<int> &hidden_blocks) = 0;

protected:
  mf_parameter param;
  int nr_bins;
  int nr_folds;
  int nr_blocks_per_fold;
  bool quiet;
  Utility util;
  double cv_error;
};

class CrossValidator : public CrossValidatorBase {
public:
    CrossValidator(mf_parameter param_, int nr_folds_, mf_problem const *prob_)
    : CrossValidatorBase(param_, nr_folds_), prob(prob_) {};
    
    double do_cv1(std::vector<int> &hidden_blocks) {
      double err = 0;
      fpsg(prob, nullptr, param, hidden_blocks, &err);
      return err;
    }
private:
    mf_problem const *prob;
};

mf_model* mf_train_with_validation(mf_problem const *tr,
                                   mf_problem const *va,
                                   mf_parameter param) {
  if (!check_parameter(param))
    return nullptr;

  std::shared_ptr<mf_model> model = nullptr;
  
  if (param.fun != P_L2_MFOC) {
    // Use stochastic gradient method
    model = fpsg(tr, va, param);
  } else {
    // Use coordinate descent method
    HML_LOG_FATAL << "Currently coordinate descent is not supported";
  }

  mf_model *model_ret = new mf_model;

  model_ret->fun = model->fun;
  model_ret->m = model->m;
  model_ret->n = model->n;
  model_ret->k = model->k;
  model_ret->b = model->b;

  model_ret->P = model->P;
  model->P = nullptr;

  model_ret->Q = model->Q;
  model->Q = nullptr;

  return model_ret;
}

mf_model* mf_train(mf_problem const *prob, mf_parameter param) {
  return mf_train_with_validation(prob, nullptr, param);
}

double mf_cross_validation(mf_problem const *prob, 
                           int nr_folds,
                           mf_parameter param) {
  // Two conditions lead to empty model. First, any parameter is not in its
  // supported range. Second, one-class matrix facotorization with L2-loss
  // (-f 12) doesn't support disk-level training.
  if (!check_parameter(param) || param.fun == P_L2_MFOC)
    return 0;

  CrossValidator validator(param, nr_folds, prob);

  return validator.do_cross_validation();
}

double calc_mse(mf_problem *prob, mf_model *model) {
  if (prob->nnz == 0)
    return 0;
  double loss = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:loss)
#endif
  for (long i = 0; i < prob->nnz; ++i) {
    mf_node &N = prob->R[i];
    float e = N.r - mf_predict(*model, N.u, N.v);
    loss += e*e;
  }
  return loss / prob->nnz;
}

double calc_mae(mf_problem *prob, mf_model *model) {
  if (prob->nnz == 0)
    return 0;
  double loss = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:loss)
#endif
  for (long i = 0; i < prob->nnz; ++i) {
    mf_node &N = prob->R[i];
    loss += abs(N.r - mf_predict(*model, N.u, N.v));
  }
  return loss / prob->nnz;
}

double calc_gkl(mf_problem *prob, mf_model *model) {
  if (prob->nnz == 0)
    return 0;
  double loss = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:loss)
#endif
  for (long i = 0; i < prob->nnz; ++i) {
    mf_node &N = prob->R[i];
    float z = mf_predict(*model, N.u, N.v);
    loss += N.r * log(N.r / z) - N.r + z;
  }
  return loss / prob->nnz;
}

double calc_logloss(mf_problem *prob, mf_model *model) {
  if (prob->nnz == 0)
    return 0;
  double logloss = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:logloss)
#endif
  for(long i = 0; i < prob->nnz; ++i) {
    mf_node &N = prob->R[i];
    float z = mf_predict(*model, N.u, N.v);
    if (N.r > 0)
      logloss += log(1.0 + exp(-z));
    else
      logloss += log(1.0 + exp(z));
  }
  return logloss / prob->nnz;
}

double calc_accuracy(mf_problem *prob, mf_model *model) {
  if (prob->nnz == 0)
    return 0;
  double acc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:acc)
#endif
  for (long i = 0; i < prob->nnz; ++i) {
    mf_node &N = prob->R[i];
    float z = mf_predict(*model, N.u, N.v);
    if (N.r > 0)
      acc += z > 0 ? 1 : 0;
    else
      acc += z < 0 ? 1 : 0;
  }
  return acc / prob->nnz;
}

std::pair<double, double> calc_mpr_auc(mf_problem *prob,
                                       mf_model *model, 
                                       bool transpose) {
  int mf_node::*row_ptr;
  int mf_node::*col_ptr;
  int m = 0, n = 0;
  if (!transpose) {
    row_ptr = &mf_node::u;
    col_ptr = &mf_node::v;
    m = std::max(prob->m, model->m);
    n = std::max(prob->n, model->n);
  } else {
    row_ptr = &mf_node::v;
    col_ptr = &mf_node::u;
    m = std::max(prob->n, model->n);
    n = std::max(prob->m, model->m);
  }

  auto sort_by_id = [&] (mf_node const &lhs, mf_node const &rhs) {
    return std::tie(lhs.*row_ptr, lhs.*col_ptr) <
            std::tie(rhs.*row_ptr, rhs.*col_ptr);
  };

  std::sort(prob->R, prob->R + prob->nnz, sort_by_id);

  auto sort_by_pred = [&] (std::pair<mf_node, float> const &lhs,
      std::pair<mf_node, float> const &rhs) { 
    return lhs.second < rhs.second; 
  };

  std::vector<int> pos_cnts(m+1, 0);
  for (int i = 0; i < prob->nnz; ++i)
    pos_cnts[prob->R[i].*row_ptr+1] += 1;
  for (int i = 1; i < m+1; ++i)
    pos_cnts[i] += pos_cnts[i-1];

  int total_m = 0;
  long total_pos = 0;
  double all_u_mpr = 0;
  double all_u_auc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
reduction(+: total_m, total_pos, all_u_mpr, all_u_auc)
#endif
  for (int i = 0; i < m; ++i) {
    if (pos_cnts[i+1] - pos_cnts[i] < 1)
      continue;

    std::vector<std::pair<mf_node, float>> row(n);

    for (int j = 0; j < n; ++j) {
      mf_node N;
      N.*row_ptr = i;
      N.*col_ptr = j;
      N.r = 0;
      row[j] = std::make_pair(N, mf_predict(*model, N.u, N.v));
    }

    int pos = 0;
    std::vector<int> index(pos_cnts[i+1] - pos_cnts[i], 0);
    for (int j = pos_cnts[i]; j < pos_cnts[i+1]; ++j) {
      if (prob->R[j].r <= 0)
        continue;

      int col = prob->R[j].*col_ptr;
      row[col].first.r = prob->R[j].r;
      index[pos] = col;
      pos += 1;
    }

    if (n-pos < 1 || pos < 1)
      continue;

    ++total_m;
    total_pos += pos;

    int count = 0;
    for (int k = 0; k < pos; ++k) {
      std::swap(row[count], row[index[k]]);
      ++count;
    }
    std::sort(row.begin(), row.begin() + pos, sort_by_pred);

    double u_mpr = 0;
    double u_auc = 0;
    for (auto neg_it = row.begin() + pos; neg_it != row.end(); ++neg_it) {
      if (row[pos-1].second <= neg_it->second) {
        u_mpr += pos;
        continue;
      }

      int left = 0;
      int right = pos-1;
      while (left < right) {
        int mid = (left + right) / 2;
        if (row[mid].second > neg_it->second)
          right = mid;
        else
          left = mid+1;
      }
      u_mpr += left;
      u_auc += pos-left;
    }

    all_u_mpr += u_mpr / (n - pos);
    all_u_auc += u_auc / (n - pos) / pos;
  }

  all_u_mpr /= total_pos;
  all_u_auc /= total_m;

  return std::make_pair(all_u_mpr, all_u_auc);
}

double calc_mpr(mf_problem *prob, mf_model *model, bool transpose) {
  return calc_mpr_auc(prob, model, transpose).first;
}

double calc_auc(mf_problem *prob, mf_model *model, bool transpose) {
  return calc_mpr_auc(prob, model, transpose).second;
}

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_COMMON_API_H_
