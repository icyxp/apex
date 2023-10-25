#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

typedef enum{
  LION_MODE_0   =0, // L2 regularization mode
  LION_MODE_1   =1  // Decoupled weight decay mode(AdamW)
} lionMode_t;

using MATH_T = float;

template<typename T, typename FULL_T>
struct LionFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    const float beta1,
    const float beta2,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    const float lr,
    lionMode_t mode,
    const float decay)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;
    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = g[i];
          r_p[ii] = p[i];
          r_m[ii] = m[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == LION_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          //r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          MATH_T update = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          if(update<=0) update = -1;
          else update = 1;
          //MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          //MATH_T next_v_unbiased = c / beta2_correction;
          //MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          //MATH_T update = next_m_unbiased / denom;
          //MATH_T update = c;
          r_p[ii] = r_p[ii] - (lr * update);
          r_m[ii] = beta2 * r_m[ii] + (1-beta2) * r_g[ii];
        }
        else { // weight decay
          MATH_T update = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          if(update<=0) update = -1;
          else update = 1;
          //MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          //MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          //MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          //MATH_T update = next_m_unbiased / denom;
          update = update + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
          r_m[ii] = beta2 * r_m[ii] + (1-beta2) * r_g[ii];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = r_p[ii];
          m[i] = r_m[ii];
        }
      }
    }
  }
};

template<typename T, typename FULL_T>
struct LionCapturableFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    const float beta1,
    const float beta2,
    const int* step,
    const int bias_correction,
    const float epsilon,
    const float* lr,
    lionMode_t mode,
    const float decay,
    const float* inv_scale)
  {
    if(*noop_gmem == 1)
      return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;


    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == LION_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          MATH_T update  = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          if(update<=0) update = -1;
          else update = 1;
          //MATH_T update = r_v[ii];
          r_p[ii] = r_p[ii] - (*lr * update);
          r_m[ii] = beta2 * r_m[ii] + (1-beta2) * r_g[ii];
        }
        else { // weight decay
          MATH_T update = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          if(update<=0) update = -1;
          else update = 1;
          //MATH_T update = r_v[ii];
          r_p[ii] = r_p[ii] - (*lr * update);
          r_m[ii] = beta2 * r_m[ii] + (1-beta2) * r_g[ii];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = static_cast<T>(r_p[ii]);
          m[i] = static_cast<T>(r_m[ii]);
        }
      }
    }
  }
};

template<typename T, typename FULL_T>
struct LionCapturableMasterFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const int* step,
    const int bias_correction,
    const float epsilon,
    const float* lr,
    lionMode_t mode,
    const float decay,
    const float* inv_scale)
  {
    if(*noop_gmem == 1)
      return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;


    FULL_T* p_master = (FULL_T*)tl.addresses[3][tensor_loc];
    p_master += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p_master[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == LION_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          //r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          //r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          //MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
         // MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          //MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
         // MATH_T update = next_m_unbiased / denom;
          //r_p[ii] = r_p[ii] - (*lr * update);
          MATH_T update = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          if(update<=0) update = -1;
          else update = 1;
          r_p[ii] = r_p[ii] - (*lr * update);
          r_m[ii] = beta2 * r_m[ii] + (1-beta2) * r_g[ii];
        }
        else { // weight decay
          //r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          //r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          //MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          ///MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          //MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          //MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          //r_p[ii] = r_p[ii] - (*lr * update);
          MATH_T update = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          if(update<=0) update = -1;
          else update = 1;
          update = update + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
          r_m[ii] = beta2 * r_m[ii] + (1-beta2) * r_g[ii];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = static_cast<T>(r_p[ii]);
          p_master[i] = static_cast<FULL_T>(r_p[ii]);
          m[i] = static_cast<FULL_T>(r_m[ii]);
        }
      }
    }
  }
};

void multi_tensor_lion_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int mode,
  const int bias_correction,
  const float weight_decay)
{
  using namespace at;

  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  // Assume single type across p,g,m1,m2 now
  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "lion",
    multi_tensor_apply<3>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      LionFunctor<scalar_t_0, float>(),
      beta1,
      beta2,
      bias_correction1,
      bias_correction2,
      epsilon,
      lr,
      (lionMode_t) mode,
      weight_decay); )

  AT_CUDA_CHECK(cudaGetLastError());

}

void multi_tensor_lion_capturable_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale)
{
  using namespace at;

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "lion",
    multi_tensor_apply<3>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      LionCapturableFunctor<scalar_t_0, float>(),
      beta1,
      beta2,
      step.data_ptr<int>(),
      bias_correction,
      epsilon,
      lr.data_ptr<float>(),
      (lionMode_t) mode,
      weight_decay,
      inv_scale.data_ptr<float>()); )

  AT_CUDA_CHECK(cudaGetLastError());

}

void multi_tensor_lion_capturable_master_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale)
{
  using namespace at;

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "lion",
    multi_tensor_apply<4>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      LionCapturableMasterFunctor<scalar_t_0, float>(),
      beta1,
      beta2,
      step.data_ptr<int>(),
      bias_correction,
      epsilon,
      lr.data_ptr<float>(),
      (lionMode_t) mode,
      weight_decay,
      inv_scale.data_ptr<float>()); )

  AT_CUDA_CHECK(cudaGetLastError());

}

