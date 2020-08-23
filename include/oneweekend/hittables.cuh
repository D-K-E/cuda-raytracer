#ifndef HITTABLES_CUH
#define HITTABLES_CUH

#include <oneweekend/hittable.cuh>
#include <oneweekend/debug.hpp>
#include <oneweekend/external.hpp>


class Hittables : public Hittable {
    public:
        Hittable** list;
        int list_size;

    public:
        __device__ Hittables() {}
        __device__
            Hittables(Hittable** hlist, int size){
                list_size = size;
                list = hlist;
            }
        __device__ ~Hittables(){
            delete list;
        }

        __device__ bool hit(const Ray &r, float d_min, float d_max,
                HitRecord &rec) const override {
            HitRecord temp;
            bool hit_anything = false;
            float closest_far = d_max;
            for (int i = 0; i < list_size; i++) {
                Hittable* h = list[i];
                bool isHit = h->hit(r, d_min, closest_far, temp);
                if (isHit == true) {
                    hit_anything = true;
                    closest_far = temp.t;
                    rec = temp;
                }
            }
            return hit_anything;
        }
};

#endif
