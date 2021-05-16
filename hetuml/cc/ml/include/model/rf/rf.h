#ifndef __HETU_ML_MODEL_RF_RF_H_
#define __HETU_ML_MODEL_RF_RF_H_

#include "model/common/mlbase.h"
#include "model/gbdt/gbdt.h"

namespace hetu { 
namespace ml {
namespace rf {

using namespace hetu::ml::tree;
using namespace hetu::ml::gbdt;

namespace RFConf {
  // instance sampling ratio
  static const std::string INS_SP_RATIO = tree::TreeConf::INS_SP_RATIO;
  static const float DEFAULT_INS_SP_RATIO = 0.3;
  // feature sampling ratio
  static const std::string FEAT_SP_RATIO = tree::TreeConf::FEAT_SP_RATIO;
  static const float DEFAULT_FEAT_SP_RATIO = 0.3;
  // number of training rounds
  static const std::string NUM_ROUND = gbdt::GBDTConf::NUM_ROUND;
  static const int DEFAULT_NUM_ROUND = 100;
  // whether to use majority 
  static const std::string IS_MAJORITY_VOTING = \
    gbdt::GBDTConf::IS_MAJORITY_VOTING;
  static const bool DEFAULT_IS_MAJORITY_VOTING = true;

  // static std::vector<std::string> meaningful_keys() {
  //   return gbdt::GBDTConf::meaningful_keys();
  // }

  static Args default_args() {
    Args args = gbdt::GBDTConf::default_args();
    args[INS_SP_RATIO] = std::to_string(DEFAULT_INS_SP_RATIO);
    args[FEAT_SP_RATIO] = std::to_string(DEFAULT_FEAT_SP_RATIO);
    args[NUM_ROUND] = std::to_string(DEFAULT_NUM_ROUND);
    args[IS_MAJORITY_VOTING] = std::to_string(DEFAULT_IS_MAJORITY_VOTING);
    return args;
  }

  static Args AddRFArgs(const Args& args) {
    Args res = args;
    if (res.find(INS_SP_RATIO) == res.end()) {
      res[INS_SP_RATIO] = std::to_string(DEFAULT_INS_SP_RATIO);
    } else {
      float ins_sp_ratio = argparse::Get<float>(res, INS_SP_RATIO);
      ASSERT_LT(ins_sp_ratio, 1) 
        << "Invalid instance sampling ratio for RF: " << ins_sp_ratio;
    }
    if (res.find(FEAT_SP_RATIO) == res.end()) {
      res[FEAT_SP_RATIO] = std::to_string(DEFAULT_FEAT_SP_RATIO);
    } else {
      float feat_sp_ratio = argparse::Get<float>(res, FEAT_SP_RATIO);
      ASSERT_LT(feat_sp_ratio, 1) 
        << "Invalid feature sampling ratio for RF: " << feat_sp_ratio;
    }
    if (res.find(NUM_ROUND) == res.end()) {
      res[NUM_ROUND] = std::to_string(DEFAULT_NUM_ROUND);
    }
    if (res.find(IS_MAJORITY_VOTING) == res.end()) {
      res[IS_MAJORITY_VOTING] = std::to_string(DEFAULT_IS_MAJORITY_VOTING);
    } else {
      bool is_majority_voting = argparse::GetBool(
        res, GBDTConf::IS_MAJORITY_VOTING);
      ASSERT(is_majority_voting) << "RF must set majority voting as true";
    }
    return res;
  }
};

template <typename Val>
class RF : public gbdt::GBDT<Val> {
public:
  inline RF(const Args& args = {})
  : GBDT<Val>{ RFConf::AddRFArgs(args) } {}
  inline const char* name() const override { return "RandomForest"; }
};

} // namespace rf
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_RF_RF_H_
