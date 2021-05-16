#ifndef __HETU_ML_MODEL_MF_COMMON_COMMON_H_
#define __HETU_ML_MODEL_MF_COMMON_COMMON_H_

#include <math.h>

namespace hetu { 
namespace ml {
namespace mf {

const int kALIGNByte = 32;
const int kALIGN = kALIGNByte / sizeof(float);

enum {P_L2_MFR=0, P_L1_MFR=1, P_KL_MFR=2, P_LR_MFC=5, P_L2_MFC=6, P_L1_MFC=7,
      P_ROW_BPR_MFOC=10, P_COL_BPR_MFOC=11, P_L2_MFOC=12};
enum {RMSE=0, MAE=1, GKL=2, LOGLOSS=5, ACC=6, ROW_MPR=10, COL_MPR=11,
      ROW_AUC=12, COL_AUC=13};

struct mf_node {
  int u;
  int v;
  float r;
};

struct mf_problem {
  int m;
  int n;
  long nnz = 0;
  mf_node* R;
};

class BlockBase {
public:
  virtual bool move_next() { return false; };
  virtual mf_node* get_current() { return nullptr; }
  virtual void reload() {};
  virtual void free() {};
  virtual long get_nnz() { return 0; };
  virtual ~BlockBase() {};
};

class Block : public BlockBase {
public:
  Block() : first(nullptr), last(nullptr), current(nullptr) {};
  Block(mf_node *first_, mf_node *last_)
      : first(first_), last(last_), current(nullptr) {};
  bool move_next() { return ++current != last; }
  mf_node* get_current() { return current; }
  void tie_to(mf_node *first_, mf_node *last_) {
    first = first_;
    last = last_;
  }
  void reload() { current = first-1; };
  long get_nnz() { return last-first; };

private:
  mf_node* first;
  mf_node* last;
  mf_node* current;
};

struct mf_parameter {
  int fun;
  int k;
  int nr_workers;
  int nr_threads;
  int nr_bins;
  int nr_iters;
  float lambda_p1;
  float lambda_p2;
  float lambda_q1;
  float lambda_q2;
  float eta;
  float alpha;
  float c;
  bool do_nmf;
  bool quiet;
};

struct mf_model {
  int fun;
  int m;
  int n;
  int k;
  float b;
  float* P;
  float* Q;
};

float mf_predict(const mf_model& model, int u, int v) {
  if (u < 0 || u >= model.m || v < 0 || v >= model.n)
    return model.b;

  float *p = model.P + (long) u * model.k;
  float *q = model.Q + (long) v * model.k;

  float z = std::inner_product(p, p + model.k, q, 0.0f);

  if (std::isnan(z))
    z = model.b;

  if (model.fun == P_L2_MFC ||
      model.fun == P_L1_MFC ||
      model.fun == P_LR_MFC)
    z = z > 0.0f? 1.0f: -1.0f;

  return z;
}

} // namespace mf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_MF_COMMON_COMMON_H_
