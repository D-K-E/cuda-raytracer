#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include <nextweek/aabb.cuh>
#include <nextweek/ray.cuh>

class Material;

/**
 @brief Records information about the ray hit.

 The information recorded in here is later used in
 scattering code. Basically what we have here should
 be sufficient for evaluating a bidirectional surface
 distribution reflection function.

 @param t distance to hit point location
 @param p the point that is hit by the ray
 @param normal the normal to the hit point.
 @param mat_ptr material attributed to hit point.
 @param u normalized screen space u coordinate
 @param v normalized screen space v coordinate
 @param front_face check which indicates the direction
 of the normal
 */
struct HitRecord {
  float t;
  Point3 p;
  Vec3 normal;
  Material *mat_ptr;
  float u, v;
  bool front_face;

  /**
    @brief check if ray hits the front side of the object

    We check the angle between the incoming ray and
    hit point. When vectors point in same direction (not
    necessarily parallel)
    their dot product is positive.

    @param r incoming ray
    @param norm the surface normal of the hit point.
   */
  __host__ __device__ void set_front_face(const Ray &r,
                                 const Vec3 &norm) {
    front_face = dot(r.direction(), norm) < 0.0f;
    normal = front_face ? norm : -norm;
  }
};

/**
  @brief A Scene Object that can be hit by ray.

  Hittable is the base class for all objects that are going
  to be traversed and hit by our algorithm. As opposed to
  those that are for sure not going to be hit, like sky,
  sun, etc.

  @param center is not necessary but represents
  the center of the bounding box of the object.

  @param mat_ptr material of the object that contributes to
  the bsdf.
 */
class Hittable {
public:
  Material *mat_ptr;
  Point3 center;

public:
  /**
    @brief determine if this object is hit by the incoming
    ray

    @param r incoming ray.
    @param d_min minimum distance.
    @param d_max maximum distance.
    @param rec the record object that would hold the
    parameters required by the scattering function later on.
   */
  __host__ __device__ virtual bool hit(const Ray &r, float d_min,
                              float d_max,
                              HitRecord &rec) const = 0;

  /**
    @brief determine the bounding box of this object.

    Compute the bounding box of a given object.
    For moving objects the time information is necessary
    since their bounding box covers the whole space between
    their traveled time period.

    @param t0 time0
    @param t1 time1
   */
  __host__ __device__ virtual bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const = 0;
};

#endif
