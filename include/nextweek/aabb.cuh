#ifndef AABB_CUH
#define AABB_CUH

#include <nextweek/external.hpp>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

/**
  @brief axis aligned bounding box

  A simple acceleration structure that helps us to organize
  the scene in what is know as an axis aligned bounding box.
  A good description is given in <a
  href="https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies/axis-alignedboundingboxes(aabbs)">P.
  Shirley's text</a>.
  The main idea is the following for objects that inside the
  same Aabb, when a
  hit occurs, their hit distance overlaps. We check that to
  ensure we have a
  hit or not. Basically instead of checking all objects
  inside the Aabb, by
  simply checking maximum and minimum point we check if a
  hit occurred.
 */
class Aabb {
public:
  __host__ __device__ Aabb() {}
  __host__ __device__ Aabb(const Point3 &a,
                           const Point3 &b) {
    _min = a;
    _max = b;
    cent = (_max - _min) / 2.0;
    volume = compute_box_volume(a, b);
  }

  __host__ __device__ Point3 min() const { return _min; }
  __host__ __device__ Point3 max() const { return _max; }
  __host__ __device__ double
  compute_box_volume(const Point3 &a,
                     const Point3 &b) const {
    // from
    // Agoston (Max), Computer Graphics and Geometric
    // Modeling: Mathematics, London, 2005. ISBN :
    // 1-85233-817-2. page 256
    Vec3 minv(dfmin(a.x(), b.x()), dfmin(a.y(), b.y()),
              dfmin(a.z(), b.z()));
    Vec3 maxv(dfmax(a.x(), b.x()), dfmax(a.y(), b.y()),
              dfmax(a.z(), b.z()));
    return (maxv.x() - minv.x()) * (maxv.y() - minv.y()) *
           (maxv.z() - minv.z());
  }
  __host__ __device__ bool hit(const Ray &r, double tmin,
                               double tmax) const {
    for (int a = 0; a < 3; a++) {
      double t0 = dfmin(
          (_min[a] - r.origin()[a]) / r.direction()[a],
          (_max[a] - r.origin()[a]) / r.direction()[a]);
      double t1 = dfmax(
          (_min[a] - r.origin()[a]) / r.direction()[a],
          (_max[a] - r.origin()[a]) / r.direction()[a]);
      tmin = dfmax(t0, tmin);
      tmax = dfmin(t1, tmax);
      if (tmax <= tmin)
        return false;
    }
    return true;
  }
  __host__ __device__ Point3 center() const { return cent; }

public:
  Point3 _min;
  Point3 _max;
  double volume;
  Point3 cent; // center
};

__host__ __device__ Aabb surrounding_box(Aabb b1, Aabb b2) {
  Point3 b1min = b1.min();
  Point3 b2min = b2.min();
  Point3 small(dfmin(b1min.x(), b2min.x()),
               dfmin(b1min.y(), b2min.y()),
               dfmin(b1min.z(), b2min.z()));

  Point3 b1max = b1.max();
  Point3 b2max = b2.max();

  Point3 big(dfmax(b1max.x(), b2max.x()),
             dfmax(b1max.y(), b2max.y()),
             dfmax(b1max.z(), b2max.z()));

  return Aabb(small, big);
}

#endif
