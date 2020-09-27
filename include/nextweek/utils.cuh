#ifndef UTILS_CUH
#define UTILS_CUH

// some utility functions
#include <nextweek/external.hpp>

const float PI = 3.141592653589f;

__host__ __device__ float degree_to_radian(float d) {
  return d * PI / 180.0f;
}

__host__ __device__ float dfmin(float f1, float f2) {
  return f1 < f2 ? f1 : f2;
}
__host__ __device__ float dfmax(float f1, float f2) {
  return f1 > f2 ? f1 : f2;
}

__host__ __device__ float clamp(float v, float mn,
                                float mx) {
  if (v < mn)
    return mn;
  if (v > mx)
    return mx;
  return v;
}

// rand utils

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__host__ __device__ float randf(unsigned int seed) {
  thrust::random::default_random_engine rng(seed);
  thrust::random::normal_distribution<float> dist(0.0f,
                                                  1.0f);
  return dist(rng);
}
__host__ __device__ float randf(unsigned int seed, int mn,
                                int mx) {
  thrust::random::default_random_engine rng(seed);
  thrust::random::normal_distribution<float> dist(mn, mx);
  return dist(rng);
}
__host__ __device__ int randint(unsigned int seed) {
  return static_cast<int>(randf(seed));
}
__host__ __device__ int randint(unsigned int seed, int mn,
                                int mx) {
  return static_cast<int>(randf(seed, mn, mx));
}

// imutils

std::vector<unsigned char> imread(const char *impath,
                                  int &w, int &h,
                                  int &nbChannels) {
  std::vector<unsigned char> imdata;
  unsigned char *data =
      stbi_load(impath, &w, &h, &nbChannels, 0);
  for (int k = 0; k < w * h * nbChannels; k++) {
    imdata.push_back(data[k]);
  }
  return imdata;
}
void imread(std::vector<const char *> impaths,
            std::vector<int> &ws, std::vector<int> &hs,
            std::vector<int> &nbChannels,
            std::vector<unsigned char> &imdata, int &size) {
  for (int i = 0; i < impaths.size(); i++) {
    int w, h, c;
    unsigned char *data =
        stbi_load(impaths[i], &w, &h, &c, 0);
    ws.push_back(w);
    hs.push_back(h);
    nbChannels.push_back(c);
    size += w * h * c;
    for (int k = 0; k < w * h * c; k++) {
      imdata.push_back(data[k]);
    }
  }
}

template <typename T>
__host__ __device__ void swap(T *&hlist, int index_h1,
                              int index_h2) {
  T temp = hlist[index_h1];
  hlist[index_h1] = hlist[index_h2];
  hlist[index_h2] = temp;
}

template <typename T>
__host__ __device__ void odd_even_sort(T *&hlist,
                                       int list_size) {
  bool sorted = false;
  while (!sorted) {
    sorted = true;
    for (int i = 1; i < list_size - 1; i += 2) {
      if (hlist[i] > hlist[i + 1]) {
        swap(hlist, i, i + 1);
        sorted = false;
      }
    }
    for (int i = 0; i < list_size - 1; i += 2) {
      if (hlist[i] > hlist[i + 1]) {
        swap(hlist, i, i + 1);
        sorted = false;
      }
    }
  }
}

template <class T, class U>
__host__ __device__ void odd_even_sort(T *&hlist, U *&ulst,
                                       int list_size) {
  bool sorted = false;
  while (!sorted) {
    sorted = true;
    for (int i = 1; i < list_size - 1; i += 2) {
      if (hlist[i] > hlist[i + 1]) {
        swap(hlist, i, i + 1);
        swap(ulst, i, i + 1);
        sorted = false;
      }
    }
    for (int i = 0; i < list_size - 1; i += 2) {
      if (hlist[i] > hlist[i + 1]) {
        swap(hlist, i, i + 1);
        swap(ulst, i, i + 1);
        sorted = false;
      }
    }
  }
}

// bvh related utils
// from nvidia post
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__host__ __device__ unsigned int
expandBits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ unsigned int morton3D(float x, float y,
                                          float z) {
  x = dfmin(dfmax(x * 1024.0f, 0.0f), 1023.0f);
  y = dfmin(dfmax(y * 1024.0f, 0.0f), 1023.0f);
  z = dfmin(dfmax(z * 1024.0f, 0.0f), 1023.0f);
  unsigned int xx = expandBits((unsigned int)x);
  unsigned int yy = expandBits((unsigned int)y);
  unsigned int zz = expandBits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

#endif
