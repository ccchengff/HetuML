#ifndef __HETU_ML_MODEL_TREE_SPLIT_H_
#define __HETU_ML_MODEL_TREE_SPLIT_H_

#include "model/common/argparse.h"

namespace hetu { 
namespace ml {
namespace tree {

class SplitEntry {
public:

  SplitEntry(): fid(-1), gain(0.0f) {}

  SplitEntry(int fid, float gain): fid(fid), gain(gain) {}

  virtual ~SplitEntry() {}

  virtual int FlowTo(float x) const = 0;

  virtual int DefaultTo() const = 0;

  inline bool is_empty() const { return fid == -1; }

  inline bool NeedReplace(float gain) const {
    return this->gain < gain;
  }
  
  inline bool NeedReplace(const SplitEntry& other) const {
    return this->gain < other.gain;
  }

  virtual SplitEntry* copy() const = 0;

  inline int get_fid() const { return fid; }

  inline void set_fid(int fid) { this->fid = fid; }

  inline float get_gain() const { return gain; }

  inline void set_gain(float gain) { this->gain = gain; }

  virtual void Print(std::ostream& os) const = 0;

  virtual std::string ToString() const = 0;

  virtual void FromString(const std::string& str) = 0;

  friend std::ostream& operator<<(std::ostream& os, const SplitEntry& split) {
    os << split.ToString();
    return os;
  }

protected:
  int fid;
  float gain;
};

class SplitPoint : public SplitEntry {
public:
  SplitPoint(): SplitEntry(), value(0.0f) {}

  SplitPoint(int fid, float value, float gain)
  : SplitEntry(fid, gain), value(value) {}

  ~SplitPoint() {}

  int FlowTo(float x) const override { return x < this->value ? 0 : 1; }

  int DefaultTo() const override { return this->value > 0.0f ? 0 : 1; }

  inline SplitEntry* copy() const override {
    return new SplitPoint(this->fid, this->value, this->gain);
  }

  inline float get_value() const { return this->value; }

  inline void set_value(float value) { this->value = value; }

  void Print(std::ostream& os) const override {
    os << "{ fid[" << this->fid << "], value[" << this->value 
      << "], gain[" << this->gain << "] }";
  }

  std::string ToString() const override {
    std::ostringstream os;
    Print(os);
    return os.str();
  }

  void FromString(const std::string& str) override {
    size_t start = 0, end = 0;
    start = argparse::GetOffset(str, "fid[", end + 1);
    end = argparse::GetOffset(str, "]", start + 1) - 1;
    this->fid = argparse::Parse<int>(str.substr(start, end - start));
    start = argparse::GetOffset(str, "value[", end + 1);
    end = argparse::GetOffset(str, "]", start + 1) - 1;
    this->value = argparse::Parse<float>(str.substr(start, end - start));
    start = argparse::GetOffset(str, "gain[", end + 1);
    end = argparse::GetOffset(str, "]", start + 1) - 1;
    this->gain = argparse::Parse<float>(str.substr(start, end - start));
  }

private:
  float value;
};

} // namespace tree
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_TREE_SPLIT_H_
