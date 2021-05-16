#ifndef __HETU_ML_PS_PSF_PSF_PSFUNCTION_H_
#define __HETU_ML_PS_PSF_PSF_PSFUNCTION_H_

enum PsfType {
  /* unary ops */
  DensePush,
  DensePull,
  SparsePush,
  SparsePull,
  Nnz,
  Norm2,
  /* Matrix ops */
  PushCols,
  PullCols,
  /* binary ops */
  AddTo,
  Minus,
  Dot,
  Axpy,
  PushPull,
  InitAllZeros,
  Other
};

#endif // __HETU_ML_PS_PSF_PSF_PSFUNCTION_H_
