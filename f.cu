#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <torch/extension.h>

// #define SEQ_LEN 2048        
// #define D_MODEL 2048
// #define NUM_HEADS 8

template<int HEAD_DIM, int Br, int Bc , int NUM_HEADS>
__global__ void forwardFlashAttention(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ out,
    float* __restrict__ L_global, // Added for backward pass tracking
    int L, int D,
    int seqlen, int headdim 
)
{
    __align__(16) extern __shared__ __half smem[]; 

    __half* smenA   = smem;                              
    __half* smenB   = smenA   + 16 * 16;                 
    __half* qshared = smenB   + 16 * 8;                  
    __half* kshared = qshared + Br * Bc;                 
    __half* vshared = kshared + Bc * Bc;                 
    float* output  = (float*)(vshared + Br * HEAD_DIM); 
    float* m = (float*)(output + Br * Bc);              
    float* l = m + Br;                                   
    float* alpha_s = l + Br;                             
    float* beta_s  = alpha_s + Br;         
    
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
                            
    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;

    const int tid     = threadIdx.x;

    const int lane  = tid % 32;
    const int group = lane / 4;

    const long long base = (long long)batchid * NUM_HEADS * seqlen * headdim +
                           (long long)headid  * seqlen * headdim;

    const long long stats_base = (long long)batchid * NUM_HEADS * seqlen +
                                 (long long)headid  * seqlen;

    const __half* Qptr = Q + base;
    const __half* Kptr = K + base;
    const __half* Vptr = V + base;
          __half* optr = out + base;
          float* Lptr = L_global + stats_base;

    for(int i = tid ; i < 64 ; i += blockDim.x)
        m[i] = -FLT_MAX , l[i] = 0.f;

    __syncthreads();

    const int rowtileid = tileid;
    const int total  = seqlen / Bc;  
    const int colitr = headdim / Bc;
    
    for(int rowid = 0 ; rowid < total ; rowid++)
    {
        for(int i = tid ; i < Br * Bc ; i += blockDim.x)
            output[i] = 0.f;
        __syncthreads();

        for(int col = 0 ; col < colitr ; col++)
        {
            for(int i = tid ; i < Br * Bc / 8; i += blockDim.x)
            {
                int r = i / 4;  
                int c = i % 4;

                *reinterpret_cast<float4*>(&qshared[r * 32 + c * 8]) = 
                    *reinterpret_cast<const float4*>(&Qptr[rowtileid * headdim * Br + col * 32 + r * headdim + c * 8]);
            }

            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
            {
                int r = i / 32;
                int c = i % 32;

                kshared[c * 32 + r] = Kptr[rowid * Bc * headdim + col * 32 + r * headdim + c];
            }

            // MMA block
            {
                for(int tr = 0 ; tr < Br / 16 ; tr++) 
                {
                    for(int col = 0 ; col < Bc / 8 ; col++)
                    {
                        float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                        for(int tc = 0 ; tc < Bc / 16 ; tc++) 
                        {
                            for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                            {
                                int r = i / 2;
                                int c = i % 2;

                                *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) = 
                                    *reinterpret_cast<const float4*>(&qshared[tr * 16 * 32 + tc * 16 + r * 32 + c * 8]);
                            }

                            for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                            {
                                int r = i;
                                int c = i % 1;
                                
                                *reinterpret_cast<float4*>(&smenB[r * 8 + c]) = 
                                    *reinterpret_cast<const float4*>(&kshared[tc * 16 * 32 + col * 8 + r * 32 + c]);
                            }

                            __syncthreads();

                            const int col0 = (lane % 4) * 2; 
                            const int col1 = col0 + 8;

                            uint32_t a_frag[4];
                            a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[group       * 16 + col0]);
                            a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                            a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[group       * 16 + col1]);
                            a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                            uint32_t b_frag[2];

                            const int r0 = (lane % 4) * 2;
                            const int r1 = r0 + 8;

                            b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));
                            b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                            __syncthreads();
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                "{%0 , %1 , %2 , %3},"
                                "{%4 , %5 , %6 , %7},"
                                "{%8 , %9},"
                                "{%10 , %11 , %12 , %13};"
                                    :"=f"(d1) , "=f"(d2) , "=f"(d3) , "=f"(d4)
                                    : "r"(a_frag[0]), "r"(a_frag[1]),
                                    "r"(a_frag[2]), "r"(a_frag[3]),
                                    "r"(b_frag[0]), "r"(b_frag[1])
                                    ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                            );
                        }

                        const int r0 = group;
                        const int r1 = r0 + 8;
                        const int c0 = (lane % 4) * 2;
                        const int c1 = c0 + 1 ;

                        output[(tr * 16 + r0) * 32 + col * 8 + c0] += d1 * scale;
                        output[(tr * 16 + r0) * 32 + col * 8 + c1] += d2 * scale;
                        output[(tr * 16 + r1) * 32 + col * 8 + c0] += d3 * scale;
                        output[(tr * 16 + r1) * 32 + col * 8 + c1] += d4 * scale;
                    }
                }
            }
        }

        // Load V into shared memory
        for(int i = tid; i < Br * headdim / 8; i += blockDim.x)
        {
            *reinterpret_cast<float4*>(&vshared[i * 8]) =
            *reinterpret_cast<const float4*>(&Vptr[rowid * 32 * headdim + i * 8]);
        }

        __syncthreads();

        if(tid < 64)
        {
            float m_tile = -FLT_MAX;
            float l_tile = 0.f;

            for(int c = 0 ; c < 32 ; c++)
                m_tile = fmaxf(m_tile , output[tid * 32 + c]);
            
            for(int c = 0 ; c < 32 ; c++)
            {
                output[tid * 32 + c] = expf(output[tid * 32 + c] - m_tile);
                l_tile += output[tid * 32 + c]; 
            }

            float m_old  =   m[tid];
            float m_new  = fmaxf(m_old , m_tile);
            alpha_s[tid] = expf(m_old  - m_new);
            beta_s[tid]  = expf(m_tile - m_new);

            l[tid] = alpha_s[tid] * l[tid] + beta_s[tid] * l_tile;
            m[tid] = m_new;
        }

        __syncthreads();

        const int itr = headdim / 8;    
        const int rowitr = Br / 16; 

        for(int iter = 0 ; iter < rowitr ; iter++)
        {
            for(int colitr = 0 ; colitr < itr ; colitr++)
            {
                float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;
                
                for(int cc = 0 ; cc < 2 ; cc++)
                {
                    for(int i = tid ; i < 256 ; i += blockDim.x)
                    {
                        int r = i / 16;
                        int c = i % 16;

                        smenA[r * 16 + c] = __float2half_rn(output[iter * 16 * 32 + cc * 16 + r * 32 + c]);
                    }

                    for(int i = tid ; i < 128 ; i += blockDim.x)
                    {
                        int r = i / 8;
                        int c = i % 8;

                        smenB[r * 8 + c] = vshared[(cc * 16 + r) * headdim + colitr * 8 + c];
                    }
                    __syncthreads();

                    const int col0 = (lane % 4) * 2;
                    const int col1 = col0 + 8;

                    uint32_t a_frag[4];
                    a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col0]);
                    a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col0]);
                    a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + col1]);
                    a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + col1]);

                    const int r0 = (lane % 4) * 2;
                    const int r1 = r0 + 8;

                    uint32_t b_frag[2];
                    b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));
                    b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));

                    __syncthreads();
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},"
                        "{%4,%5,%6,%7},"
                        "{%8,%9},"
                        "{%10,%11,%12,%13};"
                            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                            : "r"(a_frag[0]), "r"(a_frag[1]),
                            "r"(a_frag[2]), "r"(a_frag[3]),
                            "r"(b_frag[0]), "r"(b_frag[1])
                            ,"f"(d1),"f"(d2),"f"(d3),"f"(d4)
                        );
                }

                const int roww = rowtileid * Br * headdim + iter * 16 * headdim;
                const int colbase = colitr * 8;

                const int rr0 = group;
                const int rr1 = rr0 + 8;
                const int cc0 = (lane % 4) * 2;
                const int cc1 = cc0 + 1;

                float aa = __half2float(optr[roww + colbase + rr0 * headdim + cc0]);
                float ab = __half2float(optr[roww + colbase + rr0 * headdim + cc1]);
                float ac = __half2float(optr[roww + colbase + rr1 * headdim + cc0]);
                float ad = __half2float(optr[roww + colbase + rr1 * headdim + cc1]);

                aa = alpha_s[actual_row0] * aa + beta_s[actual_row0] * d1;
                ab = alpha_s[actual_row0] * ab + beta_s[actual_row0] * d2;
                ac = alpha_s[actual_row1] * ac + beta_s[actual_row1] * d3;
                ad = alpha_s[actual_row1] * ad + beta_s[actual_row1] * d4;

                optr[roww + colbase + rr0 * headdim + cc0] = __float2half(aa);
                optr[roww + colbase + rr0 * headdim + cc1] = __float2half(ab);
                optr[roww + colbase + rr1 * headdim + cc0] = __float2half(ac);
                optr[roww + colbase + rr1 * headdim + cc1] = __float2half(ad);
            }
        }
    }   

    for(int idx = tid; idx < Br * headdim; idx += blockDim.x)
    {
        int r = idx / headdim;
        int c = idx % headdim;
        float linv = __fdividef(1.0f, l[r]);
        int global_idx = rowtileid * Br * headdim + r * headdim + c;
        optr[global_idx] = __float2half(__half2float(optr[global_idx]) * linv);
    }

    __syncthreads();

  
    if (tid < Br) {
        int global_row_idx = rowtileid * Br + tid;
        if (global_row_idx < seqlen) {
            Lptr[global_row_idx] = m[tid] + logf(l[tid]);
        }
    }
}


template<int Br , int Bc , int NUM_HEADS>
__global__ void backwardFlashAttention(
    const __half* __restrict__ Q,         // [seq_len, headdim]
    const __half* __restrict__ K,         // [seq_len, headdim]
    const __half* __restrict__ V,         // [seq_len, headdim]
    const __half* __restrict__ O,
    const float* __restrict__ L_,         // Log-sum-exp statistics from Forward Pass [seq_len]
    __half* __restrict__ score,           // Shared/Global intermediate buffer
    __half* __restrict__ dL_dout,         // Incoming gradients from output [seq_len, headdim]
    __half* __restrict__ dL_dQ,           // Output Gradient Query [seq_len, headdim]
    __half* __restrict__ dL_dK,           // Output Gradient Key [seq_len, headdim]
    __half* __restrict__ dL_dV,           // Output Gradient Value [seq_len, headdim]
    int seq_len,
    int headdim
)

{
    __align__(16) extern __shared__ __half smen[];

    // Br = 32; 
    // Bc = 32 or 64

    __half* smenA    = (__half*)smen;                 
    __half* smenB    = smenA   + (16 * 16);          

    __half* qshared  = smenB   + (16 * 8);           
    __half* kshared  = qshared + (Br * Bc);          
    __half* vshared  = kshared + (Bc * Bc);           
    __half* dl_dout  = vshared + (Bc * Bc);    
    __half* dl_score = dl_dout + (Br * Bc);    

    __half* scores   = dl_score + (Br * Bc);           
    __half* dl_ds    = scores   + (Br * Bc);           

    float* dot       = (float*)(dl_ds + (Br * Bc));
    float* L         = (dot + Br;)    

    const int batchid = blockIdx.x;
    const int headid  = blockIdx.y;
    const int tileid  = blockIdx.z;
    const int tid     = threadIdx.x;

    const long long base = (long long)batchid * seq_len * headdim * NUM_HEADS +
                            (long long)headid * seq_len * headdim;

    const __half* Qptr = Q + base;
    const __half* Kptr = K + base;
    const __half* Vptr = V + base;
    const __half* Optr = O + base;

    const __half* outP = dL_dout + base;

    const int lane  = tid % 32;
    const int group = lane / 4;

    const int rowtileid = tileid;
    const int total     = seq_len / Bc;  // whether Bc can be 32 or 64 , 1024 / 32 = 32 or 1024 / 64 = 16
    const int rowINitr  = headdim / Bc;  // even though its kinda obvious we are covering a row all columns at once just a safe sanity check

    const int itr = headdim / Bc;

    if(tid < Br)
        L[tid] = L_[rowtileid * Br + tid];

    for(int col = 0 ; col < itr ; col++)
    {
        for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
        {
            int r = i / (Bc / 8);
            int c = i % (Bc / 8);

            *reinterpret_cast<float4*>(&dl_dout[r * Bc + c * 8]) = 
                *reinterpret_cast<float4*>(&outP[rowtileid * Br * headdim + col * Bc + r * headdim ; c * 8]);
        }

        for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
        {
            // we will use qshared for out(O)
            int r = i / (Bc / 8);
            int c = i % (Bc / 8);

            *reinterpret_cast<float4*>(&qshared[r * Bc + c * 8]) = 
                *reinterpret_cast<float4*>(&Optr[rowtileid * Br * headdim + col * Bc + r * headdim ; c * 8]);
        }

        __syncthreads();

        if(tid < Br)
        {
            float sum = 0.f;
            for(int k = 0; k < Bc; k++)
                sum += __half2float(dl_dout[tid * Bc + k]) *
                    __half2float(qshared[tid * Bc + k]);
            dot[rowtileid * Br + tid] += sum;   // accumulate over col tiles
        }

        __syncthreads();
    }

    for(int rowid = 0 ; rowid < total ; rowid++)
    {
        // we need Q @ K.T
        for(int colitr = 0 ; colitr < rowINitr ; colitr++)
        {
            // loading Q (vector float 4 load for half -- 8 elements at once)
            for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
            {
                int r = i / (Bc / 8);
                int c = i % (Bc / 8);

                *reinterpret_cast<float4*>(&qshared[r * Bc + c * 8]) =
                    *reinterpret_cast<float4*>(&Qptr[rowtileid * Br * headdim + colitr * Bc + r * headdim + c * 8]);
            }

            // cannot implement vector loading cuz we are loading in transpose form
            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
            {
                int r = i / 32;uint32_t*
                int c = i % 32;

                kshared[c * 32 + r] = Kptr[rowid * Bc * headdim + colitr * Bc + r * headdim + c];
            }

            __syncthreads();

            // we got Q and K , size (Br , Bc) @ (Bc , Bc).T -> (Br , Bc)
            // we gonna use mma -> m16n8k16

            const int Tr   = Br / 16; 
            const int Tc   = Bc / 16;
            const int citr = Bc / 8 ;

            for(int rr = 0 ; rr < Tr ; rr++)
            {
                for(int cc = 0 ; cc < citr ; cc++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int col = 0 ; col < Tc ; col++)
                    {
                        for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                        {
                            int r = i / (16 / 8);
                            int c = i % (16 / 8);

                            *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) = 
                                *reinterpret_cast<float4>(&qshared[rr * 16 * Bc + col * 16 + r * Bc + c * 8]);
                        }

                        for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                        {
                            int r = i;
                            int c = i % 1;

                            *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) = 
                                *reinterpret_cast<float4>(&kshared[cc * 8 + col * 16 * Bc + r * headdim + c * 8])
                        }

                        __syncthreads();

                        // we have smen loaded we can do mma now
                        uint32_t a_frag[4];

                        const int c0 = (lane $ 4) * 2;
                        const int c1 = c0 + 8;

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + c0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + c0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + c1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + c1]);

                        uint32_t b_frag[2];

                        const int r0 = (lane % 4) * 2;
                        const int r1 = r0 + 8;

                        b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));
                        b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));
                    
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},"
                            "{%4,%5,%6,%7},"
                            "{%8,%9},"
                            "{%10,%11,%12,%13};"
                            : "=f"(d1) , "=f"(d2) , "=f"(d3) , "=f"(d4)
                            :"r"(a_frag[0]) , "r"(a_frag[0]) , "r"(a_frag[0]) , "r"(a_frag[0]),
                             "r"(b_frag[0]) , "r"(b_frag[0]),
                             "f"(d1) , "f"(d2) , "f"(d3) , "f"(d4);
                        )

                    }

                    // now map this
                    const int r0 = group;
                    const int r1 = r0 + 8;

                    const int c0 = (lane % 4) * 2;
                    const int c1 = c0 + 1;

                    const int idx0 = rr * 16 * Bc + cc * 8 + r0 * Bc + c0;
                    const int idx1 = rr * 16 * Bc + cc * 8 + r0 * Bc + c1;
                    const int idx2 = rr * 16 * Bc + cc * 8 + r1 * Bc + c0;
                    const int idx3 = rr * 16 * Bc + cc * 8 + r1 * Bc + c1;

                    scores[idx0] = __float2half(__half2float(scores[idx0]) + d1);
                    scores[idx1] = __float2half(__half2float(scores[idx1]) + d2);
                    scores[idx2] = __float2half(__half2float(scores[idx2]) + d3);
                    scores[idx3] = __float2half(__half2float(scores[idx3]) + d4);

                    // scores shapes are Br * Bc

                }
            }
        
            // dl_score = dl_out (Br , Bc) and V.T (Bc , Bc) -> (Br , Bc)

            // loaded dl_out
            for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
            {
                int r = i / (Bc / 8);
                int c = i % (Bc / 8);

                *reinterpret_cast<float4*>(&dl_dout[r * Bc + c * 8]) =
                    *reinterpret_cast<float4*>(&outP[rowtileid * Br * headdim + colitr * Bc + r * headdim + c * 8]);
            }

            // loaded V
            for(int i = tid ; i < Bc * Bc / 8 ; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                vshared[c * Bc + r] = Vptr[rowid * Bc * headdim + colitr * Bc + r * headdim + c];
            }

            for(int rr = 0 ; rr < Tr ; rr++)
            {
                for(int cc = 0 ; cc < citr ; cc++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int col = 0 ; col < Tc ; col++)
                    {
                        for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                        {
                            int r = i / (16 / 8);
                            int c = i % (16 / 8);

                            *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) = 
                                *reinterpret_cast<float4>(&dl_dout[rr * 16 * Bc + col * 16 + r * Bc + c * 8]);
                        }

                        for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                        {
                            int r = i;
                            int c = i % 1;

                            *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) = 
                                *reinterpret_cast<float4>(&vshared[cc * 8 + col * 16 * Bc + r * headdim + c * 8])
                        }

                        __syncthreads();

                        // we have smen loaded we can do mma now
                        uint32_t a_frag[4];

                        const int c0 = (lane $ 4) * 2;
                        const int c1 = c0 + 8;

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + c0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + c0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + c1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + c1]);

                        uint32_t b_frag[2];

                        const int r0 = (lane % 4) * 2;
                        const int r1 = r0 + 8;

                        b_frag[0] = (uint32_t(__half_as_ushort(smenB[r0 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r0 + 1) * 8 + group])) << 16));
                        b_frag[1] = (uint32_t(__half_as_ushort(smenB[r1 * 8 + group])) | (uint32_t(__half_as_ushort(smenB[(r1 + 1) * 8 + group])) << 16));
                    
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},"
                            "{%4,%5,%6,%7},"
                            "{%8,%9},"
                            "{%10,%11,%12,%13};"
                            : "=f"(d1) , "=f"(d2) , "=f"(d3) , "=f"(d4)
                            :"r"(a_frag[0]) , "r"(a_frag[0]) , "r"(a_frag[0]) , "r"(a_frag[0]),
                             "r"(b_frag[0]) , "r"(b_frag[0]),
                             "f"(d1) , "f"(d2) , "f"(d3) , "f"(d4);
                        )

                    }

                    // now map this
                    const int r0 = group;
                    const int r1 = r0 + 8;

                    const int c0 = (lane % 4) * 2;
                    const int c1 = c0 + 1;

                    const int idx_d1 = rr * 16 * Bc + cc * 8 + r0 * Bc + c0;
                    const int idx_d2 = rr * 16 * Bc + cc * 8 + r0 * Bc + c1;
                    const int idx_d3 = rr * 16 * Bc + cc * 8 + r1 * Bc + c0;
                    const int idx_d4 = rr * 16 * Bc + cc * 8 + r1 * Bc + c1;

                    dl_score[idx_d1] = __float2half(__half2float(dl_score[idx_d1]) + d1);
                    dl_score[idx_d2] = __float2half(__half2float(dl_score[idx_d2]) + d2);
                    dl_score[idx_d3] = __float2half(__half2float(dl_score[idx_d3]) + d3);
                    dl_score[idx_d4] = __float2half(__half2float(dl_score[idx_d4]) + d4);

                    // dl_score shapes are Br * Bc
                }
            }
        }   
        
            __syncthreads();

            // recompute S = softmax(score) using saved L
            for(int i = tid; i < Br * Bc; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                float val = __half2float(scores[r * Bc + c]);
                float s   = expf(val / sqrtf((float)headdim) - L[rowtileid * Br + r]);
                scores[r * Bc + c] = __float2half(s);   // now scores holds S
            }

            for(int i = tid ; i < Br * Bc ; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                float val = __half2float(dl_score[r * Bc + c]);
                float s   = val - dot[r];
                s = s / sqrtf(headdim)

                dl_score[r * bc + c] = __float2half(s);

            }

            for(int i = tid ; i < Br * Bc ; i += blockDim.x)
            {
                // we need scores * dl_score
                int r = i / Bc; 
                int c = i % Bc;

                float val1 = __half2float(scores[r * Bc + c]) , val2 = __half2float(dl_score[r * Bc + c])
                val1 = val1 * val2;

                dl_ds[r * Bc + c] = __float2half(val1);
            }

            __syncthreads();
            
            // now we have proper dl/ds

            // now load Kshared
            for(int x = 0 ; x < headdim / Bc ; x++)
            {
                for(int i = tid ; i < Bc * Bc / 8 ; i += blockDim.x)
                {
                    int r = i / (Bc / 8);
                    int c = i % (Bc / 8);

                    *reinterpret_cast<float4*>(&kshared[r * Bc + c * 8]) = 
                        *reinterpret_cast<float4*>(&Kptr[])
                }
            }
            
        
    }
}
