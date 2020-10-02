#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <oneweekend/ray.cuh>

class Camera {
public:
  __device__ Camera(Vec3 orig, Vec3 target, Vec3 vup,
                    double vfov, double aspect,
                    double aperture, double focus_dist,
                    double t0 = 0.0f, double t1 = 0.0f) {
    lens_radius = aperture / 2;
    time0 = t0;
    time1 = t1;
    double theta = vfov * M_PI / 180;
    double half_height = tan(theta / 2);
    double half_width = aspect * half_height;
    origin = orig;
    w = to_unit(orig - target);
    u = to_unit(cross(vup, w));
    v = cross(w, u);
    lower_left_corner =
        origin - half_width * focus_dist * u -
        half_height * focus_dist * v - focus_dist * w;
    horizontal = 2 * half_width * focus_dist * u;
    vertical = 2 * half_height * focus_dist * v;
  }
  __device__ Ray get_ray(double s, double t,
                         curandState *lo) const {
    Vec3 rd = lens_radius * random_in_unit_disk(lo);
    Vec3 offset = u * rd.x() + v * rd.y();
    return Ray(origin + offset,
               lower_left_corner + s * horizontal +
                   t * vertical - origin - offset,
               random_double(lo, time0, time1));
  }

  Vec3 origin;
  Vec3 lower_left_corner;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 u, v, w;
  double lens_radius;
  double time0, time1;
};

#endif
