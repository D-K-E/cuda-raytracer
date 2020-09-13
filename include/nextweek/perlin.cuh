#ifndef PERLIN_CUH
#define PERLIN_CUH

#include <nextweek/external.hpp>
#include <nextweek/vec3.cuh>

class Perlin {
public:
  __device__ Perlin() {}
  __device__ Perlin(curandState *loc) {
    ranvec = new Vec3[point_count];
    for (int i = 0; i < point_count; ++i) {
      ranvec[i] = to_unit(random_vec(loc, -1.0f, 1.0f));
    }

    int *px = new int[point_count];
    perlin_generate_perm(loc, px);
    perm_x = px;
    int *py = new int[point_count];
    perlin_generate_perm(loc, py);
    perm_y = py;
    int *pz = new int[point_count];
    perlin_generate_perm(loc, pz);
    perm_z = pz;
  }

  __device__ ~Perlin() {
    delete[] ranvec;
    delete[] perm_x;
    delete[] perm_y;
    delete[] perm_z;
  }

  __device__ float noise(const Point3 &p) const {
    float u = p.x() - floor(p.x());
    float v = p.y() - floor(p.y());
    float w = p.z() - floor(p.z());
    int i = (int)(floor(p.x()));
    int j = (int)(floor(p.y()));
    int k = (int)(floor(p.z()));
    Vec3 c[2][2][2];

    for (int di = 0; di < 2; di++)
      for (int dj = 0; dj < 2; dj++)
        for (int dk = 0; dk < 2; dk++)
          c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^
                                 perm_y[(j + dj) & 255] ^
                                 perm_z[(k + dk) & 255]];

    return perlin_interp(c, u, v, w);
  }

  __device__ float turb(const Point3 &p,
                        int depth = 7) const {
    float accum = 0.0f;
    Point3 temp_p = p;
    float weight = 1.0;

    for (int i = 0; i < depth; i++) {
      accum += weight * noise(temp_p);
      weight *= 0.5;
      temp_p *= 2;
    }

    return fabs(accum);
  }

private:
  static const int point_count = 256;
  Vec3 *ranvec;
  int *perm_x;
  int *perm_y;
  int *perm_z;

  __device__ static void
  perlin_generate_perm(curandState *loc, int *p) {

    for (int i = 0; i < point_count; i++)
      p[i] = i;

    permute(p, point_count, loc);
  }

  __device__ static void permute(int *p, int n,
                                 curandState *loc) {
    for (int i = n - 1; i > 0; i--) {
      int target = random_int(loc, 0, i);
      int tmp = p[i];
      p[i] = p[target];
      p[target] = tmp;
    }
  }

  __device__ inline static double
  perlin_interp(Vec3 c[2][2][2], float u, float v,
                float w) {
    auto uu = u * u * (3 - 2 * u);
    auto vv = v * v * (3 - 2 * v);
    auto ww = w * w * (3 - 2 * w);
    auto accum = 0.0;

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
          Vec3 weight_v(u - i, v - j, w - k);
          accum += (i * uu + (1 - i) * (1 - uu)) *
                   (j * vv + (1 - j) * (1 - vv)) *
                   (k * ww + (1 - k) * (1 - ww)) *
                   dot(c[i][j][k], weight_v);
        }

    return accum;
  }
};

#endif
