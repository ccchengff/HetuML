#ifndef __HETU_ML_MODEL_MF_COMMON_SOLVER_H_
#define __HETU_ML_MODEL_MF_COMMON_SOLVER_H_

#include "common/logging.h"
#include "model/mf/common/common.h"
#include "model/mf/common/scheduler.h"
#include <memory>

namespace hetu { 
namespace ml {
namespace mf {

//--------------------------------------
//-----The base class of all solvers----
//--------------------------------------

class SolverBase {
public:
  SolverBase(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
             float *PG, float *QG, mf_model &model, mf_parameter param,
             bool &slow_only)
    : scheduler(scheduler), blocks(blocks), PG(PG), QG(QG),
      model(model), param(param), slow_only(slow_only) {}
  
  inline void run() {
    load_fixed_variables();
    while (!scheduler.is_terminated()) {
      arrange_block();
      while (block->move_next()) {
        N = block->get_current();
        p = model.P + (long) N->u * model.k;
        q = model.Q + (long) N->v * model.k;
        pG = PG + N->u * 2;
        qG = QG + N->v * 2;
        prepare_for_sg_update();
        sg_update(0, kALIGN, rk_slow);
        if (slow_only)
          continue;
        update();
        sg_update(kALIGN, model.k, rk_fast);
      }
      finalize();
    }
  }
  
  SolverBase(const SolverBase&) = delete;
  
  SolverBase& operator=(const SolverBase&) = delete;
  
  // Solver is stateless functor, so default destructor should be
  // good enough.
  virtual ~SolverBase() = default;

protected:
  virtual void prepare_for_sg_update() = 0;
  
  virtual void sg_update(int d_begin, int d_end, float rk) = 0;
  
  static void calc_z(float &z, int k, float *p, float *q) {
    z = 0;
    for (int d = 0; d < k; ++d)
      z += p[d] * q[d];
  }
  
  virtual void load_fixed_variables() {
    lambda_p1 = param.lambda_p1;
    lambda_q1 = param.lambda_q1;
    lambda_p2 = param.lambda_p2;
    lambda_q2 = param.lambda_q2;
    rk_slow = (float) 1.0 / kALIGN;
    rk_fast = (float) 1.0 / (model.k - kALIGN);
  }
  
  virtual void arrange_block() {
    loss = 0.0;
    error = 0.0;
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
  }
  
  virtual void finalize() {
    block->free();
    scheduler.put_job(bid, loss, error);
  }
  
  static float qrsqrt(float x) {
    float xhalf = 0.5f*x;
    uint32_t i;
    memcpy(&i, &x, sizeof(i));
    i = 0x5f375a86 - (i>>1);
    memcpy(&x, &i, sizeof(i));
    x = x*(1.5f - xhalf*x*x);
    return x;
  }
  
  virtual void update() { ++pG; ++qG; };

  Scheduler &scheduler;
  std::vector<BlockBase*> &blocks;
  BlockBase *block;
  float *PG;
  float *QG;
  mf_model &model;
  mf_parameter param;
  bool &slow_only;

  mf_node *N;
  float z;
  double loss;
  double error;
  float *p;
  float *q;
  float *pG;
  float *qG;
  int bid;

  float lambda_p1;
  float lambda_q1;
  float lambda_p2;
  float lambda_q2;
  float rk_slow;
  float rk_fast;
};


//--------------------------------------
//-----Real-valued MF and binary MF-----
//--------------------------------------

class MFSolver: public SolverBase {
public:
  MFSolver(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
           float *PG, float *QG, mf_model &model, mf_parameter param, 
           bool &slow_only)
    : SolverBase(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
  void sg_update(int d_begin, int d_end, float rk) {
    float eta_p = param.eta * qrsqrt(*pG);
    float eta_q = param.eta * qrsqrt(*qG);

    float pG1 = 0;
    float qG1 = 0;

    for (int d = d_begin; d < d_end; ++d) {
      float gp = -z*q[d]+lambda_p2*p[d];
      float gq = -z*p[d]+lambda_q2*q[d];

      pG1 += gp*gp;
      qG1 += gq*gq;

      p[d] -= eta_p*gp;
      q[d] -= eta_q*gq;
    }

    if (lambda_p1 > 0) {
      for (int d = d_begin; d < d_end; ++d) {
        float p1 = std::max(abs(p[d]) - lambda_p1 * eta_p, 0.0f);
        p[d] = p[d] >= 0 ? p1 : -p1;
      }
    }

    if (lambda_q1 > 0) {
      for (int d = d_begin; d < d_end; ++d) {
        float q1 = std::max(abs(q[d]) - lambda_q1 * eta_q, 0.0f);
        q[d] = q[d] >= 0 ? q1 : -q1;
      }
    }

    if (param.do_nmf) {
      for (int d = d_begin; d < d_end; ++d) {
        p[d] = std::max(p[d], 0.0f);
        q[d] = std::max(q[d], 0.0f);
      }
    }

    *pG += pG1*rk;
    *qG += qG1*rk;
  }
};

class L2_MFR : public MFSolver {
public:
  L2_MFR(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
          float *PG, float *QG, mf_model &model, mf_parameter param,
          bool &slow_only)
      : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
  void prepare_for_sg_update() {
    calc_z(z, model.k, p, q);
    z = N->r - z;
    loss += z * z;
    error = loss;
  }
};

class L1_MFR : public MFSolver {
public:
  L1_MFR(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
          float *PG, float *QG, mf_model &model, mf_parameter param,
          bool &slow_only)
      : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
  void prepare_for_sg_update() {
    calc_z(z, model.k, p, q);
    z = N->r - z;
    loss += abs(z);
    error = loss;
    if (z > 0)
      z = 1;
    else if (z < 0)
      z = -1;
  }
};

class KL_MFR : public MFSolver {
public:
  KL_MFR(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
          float *PG, float *QG, mf_model &model, mf_parameter param,
          bool &slow_only)
      : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
  void prepare_for_sg_update() {
    calc_z(z, model.k, p, q);
    z = N->r / z;
    loss += N->r * (log(z) - 1 + 1 / z);
    error = loss;
    z -= 1;
  }
};

class LR_MFC : public MFSolver {
public:
  LR_MFC(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
          float *PG, float *QG, mf_model &model, mf_parameter param,
          bool &slow_only)
      : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
  void prepare_for_sg_update() {
    calc_z(z, model.k, p, q);
    if (N->r > 0) {
      z = exp(-z);
      loss += log(1 + z);
      error = loss;
      z = z / (1 + z);
    } else {
      z = exp(z);
      loss += log(1 + z);
      error = loss;
      z = -z / (1 + z);
    }
  }
};

class L2_MFC : public MFSolver {
public:
  L2_MFC(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
          float *PG, float *QG, mf_model &model, mf_parameter param,
          bool &slow_only)
      : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
  void prepare_for_sg_update() {
    calc_z(z, model.k, p, q);
    if (N->r > 0) {
      error += z > 0 ? 1 : 0;
      z = std::max(0.0f, 1 - z);
    } else {
      error += z < 0 ? 1 : 0;
      z = std::min(0.0f, -1 - z);
    }
    loss += z*z;
  }
};

class L1_MFC : public MFSolver {
public:
  L1_MFC(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
          float *PG, float *QG, mf_model &model, mf_parameter param,
          bool &slow_only)
      : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
  void prepare_for_sg_update() {
    calc_z(z, model.k, p, q);
    if (N->r > 0) {
      loss += std::max(0.0f, 1-z);
      error += z > 0 ? 1.0f : 0.0f;
      z = z > 1 ? 0.0f : 1.0f;
    } else {
      loss += std::max(0.0f, 1+z);
      error += z < 0 ? 1.0f : 0.0f;
      z = z < -1 ? 0.0f : -1.0f;
    }
  }
};

//--------------------------------------
//------------One-class MF--------------
//--------------------------------------

class BPRSolver : public SolverBase {
public:
  BPRSolver(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
            float *PG, float *QG, mf_model &model, mf_parameter param,
            bool &slow_only, bool is_column_oriented)
    : SolverBase(scheduler, blocks, PG, QG, model, param, slow_only),
                 is_column_oriented(is_column_oriented) {}

protected:
  virtual void prepare_negative() = 0;

  static void calc_z(float &z, int k, float *p, float *q, float *w) {
    z = 0;
    for (int d = 0; d < k; ++d)
      z += p[d] * (q[d] - w[d]);
  }

  void arrange_block() {
    loss = 0.0;
    error = 0.0;
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
    bpr_bid = scheduler.get_bpr_job(bid, is_column_oriented);
  }
  
  void prepare_for_sg_update() {
    prepare_negative();
    calc_z(z, model.k, p, q, w);
    z = exp(-z);
    loss += log(1+z);
    error = loss;
    z = z/(1+z);
  }
  
  void sg_update(int d_begin, int d_end, float rk) {
    float eta_p = param.eta*qrsqrt(*pG);
    float eta_q = param.eta*qrsqrt(*qG);
    float eta_w = param.eta*qrsqrt(*wG);

    float pG1 = 0;
    float qG1 = 0;
    float wG1 = 0;

    for (int d = d_begin; d < d_end; ++d) {
      float gp = z*(w[d]-q[d]) + lambda_p2*p[d];
      float gq = -z*p[d] + lambda_q2*q[d];
      float gw = z*p[d] + lambda_q2*w[d];

      pG1 += gp*gp;
      qG1 += gq*gq;
      wG1 += gw*gw;

      p[d] -= eta_p*gp;
      q[d] -= eta_q*gq;
      w[d] -= eta_w*gw;
    }

    if (lambda_p1 > 0) {
      for (int d = d_begin; d < d_end; ++d) {
        float p1 = std::max(abs(p[d]) - lambda_p1 * eta_p, 0.0f);
        p[d] = p[d] >= 0 ? p1 : -p1;
      }
    }

    if (lambda_q1 > 0) {
      for (int d = d_begin; d < d_end; ++d) {
        float q1 = std::max(abs(w[d]) - lambda_q1 * eta_w, 0.0f);
        w[d] = w[d] >= 0 ? q1 : -q1;
        q1 = std::max(abs(q[d]) - lambda_q1 * eta_q, 0.0f);
        q[d] = q[d] >= 0 ? q1 : -q1;
      }
    }

    if (param.do_nmf) {
      for (int d = d_begin; d < d_end; ++d) {
        p[d] = std::max(p[d], 0.0f);
        q[d] = std::max(q[d], 0.0f);
        w[d] = std::max(w[d], 0.0f);
      }
    }

    *pG += pG1 * rk;
    *qG += qG1 * rk;
    *wG += wG1 * rk;
  }
  
  void finalize() {
    scheduler.put_job(bid, loss, error);
    scheduler.put_bpr_job(bid, bpr_bid);
  }

  void update() { ++pG; ++qG; ++wG; };

  bool is_column_oriented;
  int bpr_bid;
  float *w;
  float *wG;
};

class COL_BPR_MFOC : public BPRSolver
{
public:
  COL_BPR_MFOC(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
               float *PG, float *QG, mf_model &model, mf_parameter param, 
               bool &slow_only, bool is_column_oriented=true)
    : BPRSolver(scheduler, blocks, PG, QG, model, param,
                slow_only, is_column_oriented) {}
protected:
  void load_fixed_variables() {
    lambda_p1 = param.lambda_q1;
    lambda_q1 = param.lambda_p1;
    lambda_p2 = param.lambda_q2;
    lambda_q2 = param.lambda_p2;
    rk_slow = 1.0f / kALIGN;
    rk_fast = 1.0f / (model.k - kALIGN);
  }
  
  void prepare_negative() {
    int negative = scheduler.get_negative(bid, bpr_bid, model.m, model.n, 
                                          is_column_oriented);
    w = model.P + negative * model.k;
    wG = PG + negative * 2;
    std::swap(p, q);
    std::swap(pG, qG);
  }
};

class ROW_BPR_MFOC : public BPRSolver
{
public:
  ROW_BPR_MFOC(Scheduler &scheduler, std::vector<BlockBase*> &blocks,
              float *PG, float *QG, mf_model &model, mf_parameter param, 
              bool &slow_only, bool is_column_oriented = false)
    : BPRSolver(scheduler, blocks, PG, QG, model, param,
                slow_only, is_column_oriented) {}
protected:
  void prepare_negative() {
    int negative = scheduler.get_negative(bid, bpr_bid, model.m, model.n, 
                                          is_column_oriented);
    w = model.Q + negative * model.k;
    wG = QG + negative * 2;
  }
};

class SolverFactory {
public:
    static std::shared_ptr<SolverBase> get_solver(
        Scheduler &scheduler,
        std::vector<BlockBase*> &blocks,
        float *PG,
        float *QG,
        mf_model &model,
        mf_parameter param,
        bool &slow_only) {
    std::shared_ptr<SolverBase> solver = nullptr;

    switch (param.fun) {
      case P_L2_MFR:
        solver = std::shared_ptr<SolverBase>(new L2_MFR(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      case P_L1_MFR:
        solver = std::shared_ptr<SolverBase>(new L1_MFR(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      case P_KL_MFR:
        solver = std::shared_ptr<SolverBase>(new KL_MFR(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      case P_LR_MFC:
        solver = std::shared_ptr<SolverBase>(new LR_MFC(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      case P_L2_MFC:
        solver = std::shared_ptr<SolverBase>(new L2_MFC(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      case P_L1_MFC:
        solver = std::shared_ptr<SolverBase>(new L1_MFC(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      case P_ROW_BPR_MFOC:
        solver = std::shared_ptr<SolverBase>(new ROW_BPR_MFOC(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      case P_COL_BPR_MFOC:
        solver = std::shared_ptr<SolverBase>(new COL_BPR_MFOC(
          scheduler, blocks, PG, QG, model, param, slow_only));
        break;
      default:
        throw std::invalid_argument("unknown error function");
    }
    return solver;
  }
};

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_COMMON_SOLVER_H_
