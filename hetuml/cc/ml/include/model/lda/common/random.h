#ifndef __HETU_ML_MODEL_LDA_COMMON_RANDOM_H_
#define __HETU_ML_MODEL_LDA_COMMON_RANDOM_H_

#include <cmath>
#include <cstdlib>
#include <random>

namespace hetu { 
namespace ml {
namespace lda {

class Random {
 public:
	 unsigned MAX_N = ~0U;
	 unsigned RandInt() {
		 return next();
	 }

  unsigned RandInt(int n) {
    return next() % n;
  }

  float RandDouble(float x = 1.0) {
    return x * float(next()) / MAX_N;
  }

  float RandNorm(float mean = 0, float var = 1) {
    float r = randn(gen);
    return mean + r * var;
  }

  Random() : randn(0.0, 1.0) {
    x = next_prime(rand());
    y = next_prime(rand());
    z = next_prime(rand());
  }

 private:
  unsigned x, y, z;
  std::default_random_engine gen;
  std::normal_distribution<float> randn;

  unsigned next_prime(int n) {
    while (true) {
      bool prime = true;
      for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
          prime = false;
          break;
        }
      }
      if (prime) break;
      ++n;
    }
    return n;
  }

  unsigned next() {
    return x = y * x + z;
  }
};

} // namespace lda
} // namespace ml
} // namespace hetu

#endif // __HETU_ML_MODEL_LDA_COMMON_RANDOM_H_
