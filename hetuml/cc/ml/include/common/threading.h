#ifndef __HETU_ML_THREADING_H
#define __HETU_ML_THREADING_H

#ifdef _OPENMP
#include <omp.h>
#endif
#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>

#define MIN_WORKSET_SIZE (1024)

#pragma omp declare reduction(vec_double_plus : std::vector<double> :  \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(),     \
      omp_out.begin(), std::plus<double>()))                           \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

inline static void OMP_SET_NUM_THREADS(int num_thr) {
#ifdef _OPENMP
  omp_set_num_threads(num_thr);
#endif
}

inline static int OMP_GET_NUM_THREADS() {
  int ret = 1;
#ifdef _OPENMP
  #pragma omp parallel
  #pragma omp master
  ret = omp_get_num_threads();
#endif
  return ret;
}

inline static int OMP_GET_THREAD_ID() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

#endif
