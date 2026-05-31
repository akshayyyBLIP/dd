#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <torch/extension.h>

template<int HEAD_DIM, int Br, int Bc , int NUM_HEADS>
__global__ void forwardFlashAttention(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ out,
    float* __restrict__ L_global, 
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

    for(int i = tid ; i < Br ; i += blockDim.x) // we are doing row wise softmax for Br * Bc so Br rows
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
                int r = i / Bc;
                int c = i % Bc;

                kshared[c * Bc + r] = Kptr[rowid * Bc * headdim + col * Bc + r * headdim + c];
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

        if(tid < Br)
        {
            float m_tile = -FLT_MAX;
            float l_tile = 0.f;

            for(int c = 0 ; c < Bc ; c++)
                m_tile = fmaxf(m_tile , output[tid * Bc + c]);
            
            for(int c = 0 ; c < Bc ; c++)
            {
                output[tid * Bc + c] = expf(output[tid * Bc + c] - m_tile);
                l_tile += output[tid * Bc + c]; 
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
                    for(int i = tid ; i < 16 / 16 ; i += blockDim.x)
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

                int actual_row0 = iter * 16 + rr0;  
                int actual_row1 = iter * 16 + rr1;  

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
__global__ void backwardFlashAttention_DQ(
    const __half* __restrict__ Q,         // [seq_len, headdim]
    const __half* __restrict__ K,         // [seq_len, headdim]
    const __half* __restrict__ V,         // [seq_len, headdim]
    const __half* __restrict__ O,
    const float* __restrict__ L_,         // Log-sum-exp statistics from Forward Pass [seq_len]
    __half* __restrict__ score,           // Shared/Global intermediate buffer
    __half* __restrict__ dL_dout,         // Incoming gradients from output [seq_len, headdim]
    __half* __restrict__ dL_dQ,           // Output Gradient Query [seq_len, headdim]
    int seq_len,
    int headdim
)

{
    __align__(16) extern __shared__ __half smen[];

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
    float* L         = (dot + Br);    

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

    for(int col = 0 ; col < rowINitr ; col++)
    {
        for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
        {
            int r = i / (Bc / 8);
            int c = i % (Bc / 8);

            *reinterpret_cast<float4*>(&dl_dout[r * Bc + c * 8]) = 
                *reinterpret_cast<const float4*>(&outP[rowtileid * Br * headdim + col * Bc + r * headdim + c * 8]);
        }

        for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
        {
            // we will use qshared for out(O)
            int r = i / (Bc / 8);
            int c = i % (Bc / 8);

            *reinterpret_cast<float4*>(&qshared[r * Bc + c * 8]) = 
                *reinterpret_cast<const float4*>(&Optr[rowtileid * Br * headdim + col * Bc + r * headdim + c * 8]);
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
                    *reinterpret_cast<const float4*>(&Qptr[rowtileid * Br * headdim + colitr * Bc + r * headdim + c * 8]);
            }

            // cannot implement vector loading cuz we are loading in transpose form
            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
            {
                int r = i / 32;
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
                                *reinterpret_cast<const float4*>(&qshared[rr * 16 * Bc + col * 16 + r * Bc + c * 8]);
                        }

                        for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                        {
                            int r = i;
                            int c = i % 1;

                            *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) = 
                                *reinterpret_cast<const float4*>(&kshared[cc * 8 + col * 16 * Bc + r * headdim + c * 8]);
                        }

                        __syncthreads();

                        // we have smen loaded we can do mma now
                        uint32_t a_frag[4];

                        const int c0 = (lane % 4) * 2;
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
                            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                            "r"(b_frag[0]), "r"(b_frag[1]),
                            "f"(d1), "f"(d2), "f"(d3), "f"(d4)
                        ); 

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
                    *reinterpret_cast<const float4*>(&outP[rowtileid * Br * headdim + colitr * Bc + r * headdim + c * 8]);
            }

            // loaded V
            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
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
                                *reinterpret_cast<const float4*>(&dl_dout[rr * 16 * Bc + col * 16 + r * Bc + c * 8]);
                        }

                        for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                        {
                            int r = i;
                            int c = i % 1;

                            *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) = 
                                *reinterpret_cast<const float4*>(&vshared[cc * 8 + col * 16 * Bc + r * headdim + c * 8]);
                        }

                        __syncthreads();

                        // we have smen loaded we can do mma now
                        uint32_t a_frag[4];

                        const int c0 = (lane % 4) * 2;
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
                            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                            "r"(b_frag[0]), "r"(b_frag[1]),
                            "f"(d1), "f"(d2), "f"(d3), "f"(d4)
                        ); 

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
                s = s / sqrtf(headdim);

                dl_score[r * Bc + c] = __float2half(s);

            }

            for(int i = tid ; i < Br * Bc ; i += blockDim.x)
            {
                // we need scores * dl_score
                int r = i / Bc; 
                int c = i % Bc;

                float val1 = __half2float(scores[r * Bc + c]) , val2 = __half2float(dl_score[r * Bc + c]);
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
                        *reinterpret_cast<const float4*>(&Kptr[rowid * Bc * headdim + x * Bc + r * headdim + c * 8]);
                }
                __syncthreads();

                for(int Trow = 0 ; Trow < Br / 16 ; Trow++)
                {
                    for(int tcc = 0 ; tcc < Bc / 8 ; tcc++)
                    {

                        float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                        for(int tcr = 0 ; tcr < Bc / 16 ; tcr++)
                        {
                            for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                            {
                                int r = i / (16 / 8);
                                int c = i % (16 / 8);

                                *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) =    
                                    *reinterpret_cast<const float4*>(&dl_ds[Trow * 16 * Bc + tcc * 8 + tcr * 16 + r * Bc + c * 8]);
                            }

                            for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                            {
                                int r = i / (8 / 8);
                                int c = i % (8 / 8);

                                *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) =
                                    *reinterpret_cast<float4*>(&kshared[tcc * 8 + tcr * 16 * Bc + r * Bc + r * 8]);
                            }

                            uint32_t a_frag[4];
                            const int c0 = (lane % 4) * 2;
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
                                : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                                "r"(b_frag[0]), "r"(b_frag[1]),
                                "f"(d1), "f"(d2), "f"(d3), "f"(d4) 
                            ); 
                            
                        }

                        // now we need to map it 
                        const int r0 = group;
                        const int r1 = r0 + 8;

                        const int c0 = (lane % 4) * 2;
                        const int c1 = c0 + 1;

                        const int global_idx = rowtileid * Br * headdim + x * Bc;

                        const int idx_d1 = Trow * 16 * headdim + tcc * 8 + r0 * headdim + c0;
                        const int idx_d2 = Trow * 16 * headdim + tcc * 8 + r0 * headdim + c1; 
                        const int idx_d3 = Trow * 16 * headdim + tcc * 8 + r1 * headdim + c0;
                        const int idx_d4 = Trow * 16 * headdim + tcc * 8 + r1 * headdim + c1;

                        dL_dQ[idx_d1] = __float2half(__half2float(dL_dQ[global_idx + idx_d1]) + d1);
                        dL_dQ[idx_d2] = __float2half(__half2float(dL_dQ[global_idx + idx_d2]) + d2);
                        dL_dQ[idx_d3] = __float2half(__half2float(dL_dQ[global_idx + idx_d3]) + d3);
                        dL_dQ[idx_d4] = __float2half(__half2float(dL_dQ[global_idx + idx_d4]) + d4);

                    }
                }
                
            }
            
    }
}




template<int Br , int Bc , int NUM_HEADS>
__global__ void backwardFlashAttention_DK_DV(
    const __half* __restrict__ Q,         
    const __half* __restrict__ K,        
    const __half* __restrict__ V,         
    const __half* __restrict__ O,  
    const float* __restrict__ L_,         
    __half* __restrict__ score,           
    __half* __restrict__ dL_dout,        
    __half* __restrict__ dL_dK,           
    __half* __restrict__ dL_dV,           
    int seq_len,
    int headdim
)
{
    __align__(16) extern __shared__ __half smen[];

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
    float* L         = (dot + Br);    

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

    if(tid < Br)
        L[tid] = L_[rowtileid * Br + tid];

    const int rowtileid = tileid;
    const int total     = seq_len / Br;
    const int colitr    = headdim / Bc;

    for(int rowid = 0 ; rowid < total ; rowid++)
    {   
        for(int col = 0 ; col < colitr ; col++)
        {
            for(int i = tid ; i < Br * Bc / 8 ; i += blockDim.x)
            {   
                int r = i / (Bc / 8);
                int c = i % (Bc / 8);

                *reinterpret_cast<float4*>(&qshared[r * Bc + c * 8]) = 
                    *reinterpret_cast<const float4*>(&Qptr[rowid * Br * headdim + col * Bc + r * headdim + c * 8]);
            }

            for(int i = tid ; i < Bc * Bc ; i += blockDim.x)
            {
                int r = i / Bc;
                int c = i % Bc;

                kshared[c * Bc + r] = Kptr[rowtileid * Bc * headdim + col * Bc + r * headdim + c];
            }

            __syncthreads();

            const int rr = Br / 16;
            const int cc = Bc / 16;
            const int Tc = Bc / 8 ;

            for(int row = 0 ; row < rr ; row++)
            {
                for(int co = 0 ; co < Tc ; co++)
                {
                    float d1 = 0.f , d2 = 0.f , d3 = 0.f , d4 = 0.f;

                    for(int colit = 0 ; colit < cc ; colit++)
                    {
                        for(int i = tid ; i < 16 * 16 / 8 ; i += blockDim.x)
                        {
                            int r = i / (16 / 8);
                            int c = i % (16 / 8);

                            *reinterpret_cast<float4*>(&smenA[r * 16 + c * 8]) =  
                                *reinterpret_cast<const float4*>(&qshared[row * Bc * 16 + colit * 16 + r * Bc + c * 8]);
                        }

                        for(int i = tid ; i < 16 * 8 / 8 ; i += blockDim.x)
                        {
                            int r = i / (8 / 8);
                            int c = i % (8 / 8);

                            *reinterpret_cast<float4*>(&smenB[r * 8 + c * 8]) = 
                                *reinterpret_cast<const float4*>(&kshared[colit * 16 * Bc + co * 8 + r * Bc + c * 8]);
                        }

                        __syncthreads();

                        uint32_t a_frag[4];

                        const int r0 = (lane % 4) * 2;
                        const int r1 = r0 + 8;

                        a_frag[0] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + r0]);
                        a_frag[1] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + r0]);
                        a_frag[2] = *reinterpret_cast<const uint32_t*>(&smenA[ group      * 16 + r1]);
                        a_frag[3] = *reinterpret_cast<const uint32_t*>(&smenA[(group + 8) * 16 + r1]);

                        uint32_t b_frag[2];

                        const int c0 = (lane % 4) * 2;
                        const int c1 = co + 8;

                        b_frag[0] = (uint32_t(__half_as_ushort(&smenB[c0 * 8 + group])) | (uint32_t(__half_as_ushort(&smenB[(c0 + 1) * 8 + group])) << 16));
                        b_frag[1] = (uint32_t(__half_as_ushort(&smenB[c1 * 8 + group])) | (uint32_t(__half_as_ushort(&smenB[(c1 + 1) * 8 + group])) << 16));

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},"
                            "{%4,%5,%6,%7},"
                            "{%8,%9},"
                            "{%10,%11,%12,%13};"
                            :"=f"(d1) , "=f"(d2) , "=f"(d3) ,"=f"(d4)
                            :"r"(a_frag[0]) , "r"(a_frag[1]) , "r"(a_frag[2]) , "r"(a_frag[3]),
                             "r"(b_frag[0]) , "r"(b_frag[1]),
                             "f"(d1) , "f"(d2) , "f"(d3) ,"f"(d4)
                        )

                    }

                    const int r0 = group;
                    const int r1 = r0 + 8;
                    const int c0 = (lane % 4) * 2;
                    const int c1 = c0 + 1;

                    const int idx0 = row * 16 * Bc + co * 8 + r0 * Bc + co;
                    const int idx1 = row * 16 * Bc + co * 8 + r0 * Bc + c1;
                    const int idx2 = row * 16 * Bc + co * 8 + r1 * Bc + co;
                    const int idx3 = row * 16 * Bc + co * 8 + r1 * Bc + c1;

                    scores[idx0] = __float2half(__half2float(scores[idx]) + d1);
                    scores[idx1] = __float2half(__half2float(scores[idx]) + d2);
                    scores[idx2] = __float2half(__half2float(scores[idx]) + d3);
                    scores[idx3] = __float2half(__half2float(scores[idx]) + d4);
                }
            }

            // we have scores now
            // now we will do DL/DV
            
        }
    }
}



std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    auto b = q.size(0);
    auto h = q.size(1);
    auto s = q.size(2);
    auto d = q.size(3);

    auto out = torch::zeros_like(q);
    auto lse = torch::zeros({b, h, s}, q.options().dtype(torch::kFloat32));

    const int Br = 64;
    dim3 grid(b, h, s / Br);
    dim3 block(128);
    size_t smem = 48 * 1024;

    if (d == 64) {
        forwardFlashAttention<64, 64, 32, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (half*)out.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            0, 0,
            static_cast<int>(s),
            static_cast<int>(d)
        );
    } else {
        forwardFlashAttention<32, 64, 32, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (half*)out.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            0, 0,
            static_cast<int>(s),
            static_cast<int>(d)
        );
    }

    return {out, lse};
}



torch::Tensor flash_attn_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor o, torch::Tensor lse, torch::Tensor d_out
) {
    auto b = q.size(0);
    auto h = q.size(1);
    auto s = q.size(2);
    auto d = q.size(3);

    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    auto score_buf = torch::zeros({b, h, 32, static_cast<long>(d)}, q.options());

    const int Br = 32;
    dim3 grid(b, h, s / Br);
    dim3 block(128);
    size_t smem = 48 * 1024;

    if (d == 64) {
        backwardFlashAttention_DQ<32, 64, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (const half*)o.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            (half*)score_buf.data_ptr<at::Half>(),
            (half*)d_out.data_ptr<at::Half>(),
            (half*)dq.data_ptr<at::Half>(),
            static_cast<int>(s),
            static_cast<int>(d)
        );
    } else {
        backwardFlashAttention_DQ<32, 32, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (const half*)o.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            (half*)score_buf.data_ptr<at::Half>(),
            (half*)d_out.data_ptr<at::Half>(),
            (half*)dq.data_ptr<at::Half>(),
            static_cast<int>(s),
            static_cast<int>(d)
        );
    }

    return dq;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &flash_attn_forward,  "Flash Attention Forward");
    m.def("backward", &flash_attn_backward, "Flash Attention Backward");
}
