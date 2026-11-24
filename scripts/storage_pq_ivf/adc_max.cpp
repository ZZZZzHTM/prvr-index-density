// Exhaustive inner-product ADC using FAISS subquantizer tables/codes.
// No ANN pruning: maximum over every encoded vector belonging to each video.
#include <cstdint>
#include <limits>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
extern "C" void adc_max(const float* tables, const uint8_t* transposed_codes,
    const int64_t* offsets, int64_t nq, int64_t nv, int64_t ntotal,
    int m, int nthreads, float* result) {
  #pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int64_t q=0; q<nq; ++q) {
    const float* lut=tables+q*m*256;
    for(int64_t v=0; v<nv; ++v) {
      float best=-std::numeric_limits<float>::infinity();
      int64_t i=offsets[v], end=offsets[v+1];
      __m256 mx=_mm256_set1_ps(best);
      for(; i+8<=end; i+=8) {
        __m256 sums=_mm256_setzero_ps();
        for(int j=0; j<m; ++j) {
          __m128i bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(transposed_codes+j*ntotal+i));
          __m256i ids=_mm256_cvtepu8_epi32(bytes);
          sums=_mm256_add_ps(sums,_mm256_i32gather_ps(lut+j*256,ids,4));
        }
        mx=_mm256_max_ps(mx,sums);
      }
      float lanes[8];_mm256_storeu_ps(lanes,mx);
      for(int j=0;j<8;++j)best=std::max(best,lanes[j]);
      for(;i<end;++i) {
        float score=0;
        for(int j=0;j<m;++j)score+=lut[j*256+transposed_codes[j*ntotal+i]];
        best=std::max(best,score);
      }
      result[q*nv+v]=best;
    }
  }
}
