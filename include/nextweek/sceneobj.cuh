#ifndef SCENEOBJ_CUH
#define SCENEOBJ_CUH
// an interface scene object between different primitives

#include <nextweek/aarect.cuh>
#include <nextweek/cbuffer.hpp>
#include <nextweek/hittable.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/scenegroup.cuh>
#include <nextweek/sceneparam.cuh>
#include <nextweek/sceneprim.cuh>
#include <nextweek/scenetype.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/triangle.cuh>
#include <nextweek/vec3.cuh>

struct SceneObjects {
  // objects
  int *gtypes;
  int *group_ids;
  int *group_starts;
  int *group_sizes;
  int nb_group;

  // surfaces
  int *htypes;
  float *rads;
  float *p1xs, *p1ys, *p1zs;
  float *p2xs, *p2ys, *p2zs;
  float *p3xs, *p3ys, *p3zs;

  // materials
  int *mtypes;
  float *fuzzs;

  // textures
  int *ttypes;
  float *tp1xs;
  float *tp1ys;
  float *tp1zs; // solid color, checker texture
  float *tp2xs;
  float *tp2ys;
  float *tp2zs; // solid color, checker texture

  unsigned char *tdata; // image texture
  int tdata_size;       // image texture
  int *ws, *hs;         // image texture
  int *bpps;            // image texture
  int *tindices;        // image texture
  float *scales;        // noise texture
  //
  const int hlength; // objects size

  __host__ __device__ SceneObjects() : hlength(0) {}
  __host__ __device__ SceneObjects(int obj_size)
      : hlength(obj_size) {
    init_arrays();
  }
  __host__ __device__ void free() {
    cudaFree(gtypes);
    cudaFree(group_ids);
    cudaFree(group_starts);
    cudaFree(group_sizes);
    cudaFree(htypes);
    cudaFree(rads);
    cudaFree(p1xs);
    cudaFree(p1ys);
    cudaFree(p1zs);
    cudaFree(p2xs);
    cudaFree(p2ys);
    cudaFree(p2zs);
    cudaFree(p3xs);
    cudaFree(p3ys);
    cudaFree(p3zs);
    cudaFree(mtypes);
    cudaFree(fuzzs);
    cudaFree(ttypes);
    cudaFree(tp1xs);
    cudaFree(tp1ys);
    cudaFree(tp1zs);
    cudaFree(tp2xs);
    cudaFree(tp2ys);
    cudaFree(tp2zs);
    cudaFree(ws);
    cudaFree(hs);
    cudaFree(bpps);
    cudaFree(tindices);
    cudaFree(scales);
  }
  __host__ __device__ void init_arrays() {
    gtypes = new int[hlength];
    group_ids = new int[hlength];
    group_starts = new int[hlength];
    group_sizes = new int[hlength];

    //
    htypes = new int[hlength];
    rads = new float[hlength];
    p1xs = new float[hlength];
    p1ys = new float[hlength];
    p1zs = new float[hlength];
    p2xs = new float[hlength];
    p2ys = new float[hlength];
    p2zs = new float[hlength];
    p3xs = new float[hlength];
    p3ys = new float[hlength];
    p3zs = new float[hlength];
    mtypes = new int[hlength];
    fuzzs = new float[hlength];
    ttypes = new int[hlength];
    tp1xs = new float[hlength];
    tp1ys = new float[hlength];
    tp1zs = new float[hlength];
    tp2xs = new float[hlength];
    tp2ys = new float[hlength];
    tp2zs = new float[hlength];

    tdata = nullptr;
    tdata_size = 0;
    ws = new int[hlength];
    hs = new int[hlength];
    bpps = new int[hlength];
    tindices = new int[hlength];
    scales = new float[hlength];
  }
  __host__ __device__ SceneObjects(ScenePrimitive *&objs,
                                   int obj_length)
      : hlength(obj_length) {
    SceneGroup *gs = new SceneGroup(
        objs[0].group_id, obj_length, objs[0].gtype, objs);
    set_sgroups(gs, 1);
  }
  __host__ __device__ SceneObjects(SceneGroup *&objs,
                                   int nb_g,
                                   int total_prim_nb)
      : hlength(total_prim_nb) {
    set_sgroups(objs, nb_g);
  }
  __host__ __device__ void set_sgroups(SceneGroup *&objs,
                                       int nb_g) {
    init_arrays();
    nb_group = nb_g;
    gtypes = new int[nb_group];
    group_ids = new int[nb_group];
    group_starts = new int[nb_group];
    group_sizes = new int[nb_group];

    bool data_check = false;
    int i = 0;
    for (int k = 0; k < nb_group; k++) {
      SceneGroup g = objs[k];
      gtypes[k] = g.group_type;
      group_ids[k] = g.group_id;
      group_starts[k] = i;
      int prim_nb = g.group_size;
      group_sizes[k] = prim_nb;
      ScenePrimitive *sp = g.prims;
      for (int j = 0; j < prim_nb; j++) {
        ScenePrimitive prim = sp[j];
        set_object(prim, i);
        if (data_check == false && prim.data) {
          tdata = prim.data;
          data_check = true;
          tdata_size = prim.data_size;
        }
        i++;
      }
    }
  }
  __host__ __device__ void
  set_object(const ScenePrimitive &sobj, int obj_index) {
    htypes[obj_index] = sobj.htype;
    rads[obj_index] = sobj.radius;

    p1xs[obj_index] = sobj.p1x;
    p1ys[obj_index] = sobj.p1y;
    p1zs[obj_index] = sobj.p1z;

    p2xs[obj_index] = sobj.p2x;
    p2ys[obj_index] = sobj.p2y;
    p2zs[obj_index] = sobj.p2z;

    p3xs[obj_index] = sobj.p3x;
    p3ys[obj_index] = sobj.p3y;
    p3zs[obj_index] = sobj.p3z;
    //
    mtypes[obj_index] = sobj.mtype;
    fuzzs[obj_index] = sobj.fuzz;
    //
    ttypes[obj_index] = sobj.ttype;

    tp1xs[obj_index] = sobj.tp1x;
    tp1ys[obj_index] = sobj.tp1y;
    tp1zs[obj_index] = sobj.tp1z;

    tp2xs[obj_index] = sobj.tp2x;
    tp2ys[obj_index] = sobj.tp2y;
    tp2zs[obj_index] = sobj.tp2z;
    //
    ws[obj_index] = sobj.width;
    hs[obj_index] = sobj.height;
    bpps[obj_index] = sobj.bytes_per_pixel;
    tindices[obj_index] = sobj.index;
    scales[obj_index] = sobj.scale;
  }
  __host__ void to_device_group(SceneObjects &sobjs) {
    int *gts = nullptr;
    CUDA_CONTROL(upload<int>(gts, gtypes, nb_group));
    sobjs.gtypes = gts;

    int *gids = nullptr;
    CUDA_CONTROL(upload<int>(gids, group_ids, nb_group));
    sobjs.group_ids = gids;

    int *d_gstarts = nullptr;
    CUDA_CONTROL(
        upload<int>(d_gstarts, group_starts, nb_group));
    sobjs.group_starts = d_gstarts;

    int *d_gsizes = nullptr;
    CUDA_CONTROL(
        upload<int>(d_gsizes, group_sizes, nb_group));
    sobjs.group_sizes = d_gsizes;
  }
  __host__ void to_device_surface(SceneObjects &sobjs) {
    int *hts = nullptr;
    CUDA_CONTROL(upload<int>(hts, htypes, hlength));
    sobjs.htypes = hts;

    //
    float *rds;
    CUDA_CONTROL(upload<float>(rds, rads, hlength));
    sobjs.rads = rds;

    //
    float *d_p1xs;
    CUDA_CONTROL(upload<float>(d_p1xs, p1xs, hlength));
    sobjs.p1xs = d_p1xs;
    //
    float *d_p1ys;
    CUDA_CONTROL(upload<float>(d_p1ys, p1ys, hlength));
    sobjs.p1ys = d_p1ys;
    //
    float *d_p1zs;
    CUDA_CONTROL(upload<float>(d_p1zs, p1zs, hlength));
    sobjs.p1zs = d_p1zs;
    //
    float *d_p2xs;
    CUDA_CONTROL(upload<float>(d_p2xs, p2xs, hlength));
    sobjs.p2xs = d_p2xs;
    //
    float *d_p2ys;
    CUDA_CONTROL(upload<float>(d_p2ys, p2ys, hlength));
    sobjs.p2ys = d_p2ys;
    //
    float *d_p2zs;
    CUDA_CONTROL(upload<float>(d_p2zs, p2zs, hlength));
    sobjs.p2zs = d_p2zs;
    //
    float *d_p3xs;
    CUDA_CONTROL(upload<float>(d_p3xs, p3xs, hlength));
    sobjs.p3xs = d_p3xs;
    //
    float *d_p3ys;
    CUDA_CONTROL(upload<float>(d_p3ys, p3ys, hlength));
    sobjs.p3ys = d_p3ys;
    //
    float *d_p3zs;
    CUDA_CONTROL(upload<float>(d_p3zs, p3zs, hlength));
    sobjs.p3zs = d_p3zs;
  }
  __host__ void to_device_material(SceneObjects &sobjs) {
    int *d_mtypes;
    CUDA_CONTROL(upload<int>(d_mtypes, mtypes, hlength));
    sobjs.mtypes = d_mtypes;
    //
    float *d_fuzzs;
    CUDA_CONTROL(upload<float>(d_fuzzs, fuzzs, hlength));
    sobjs.fuzzs = d_fuzzs;
  }
  __host__ void to_device_texture(SceneObjects &sobjs) {
    int *d_ttypes;
    CUDA_CONTROL(upload<int>(d_ttypes, ttypes, hlength));
    sobjs.ttypes = d_ttypes;
    //
    float *d_tp1xs;
    CUDA_CONTROL(upload<float>(d_tp1xs, tp1xs, hlength));
    sobjs.tp1xs = d_tp1xs;
    //
    float *d_tp1ys;
    CUDA_CONTROL(upload<float>(d_tp1ys, tp1ys, hlength));
    sobjs.tp1ys = d_tp1ys;
    //
    float *d_tp1zs;
    CUDA_CONTROL(upload<float>(d_tp1zs, tp1zs, hlength));
    sobjs.tp1zs = d_tp1zs;
    //
    float *d_tp2xs;
    CUDA_CONTROL(upload<float>(d_tp2xs, tp2xs, hlength));
    sobjs.tp2xs = d_tp2xs;
    //
    float *d_tp2ys;
    CUDA_CONTROL(upload<float>(d_tp2ys, tp2ys, hlength));
    sobjs.tp2ys = d_tp2ys;
    //
    float *d_tp2zs;
    CUDA_CONTROL(upload<float>(d_tp2zs, tp2zs, hlength));
    sobjs.tp2zs = d_tp2zs;
  }
  __host__ SceneObjects to_device() {
    // device copy
    SceneObjects sobjs(hlength);

    to_device_group(sobjs);
    to_device_surface(sobjs);
    to_device_material(sobjs);
    to_device_texture(sobjs);
    //
    unsigned char *d_tdata;
    CUDA_CONTROL(upload<unsigned char>(d_tdata, tdata,
                                       sobjs.tdata_size));
    sobjs.tdata = d_tdata;
    //
    sobjs.tdata_size = tdata_size;
    //
    int *d_ws;
    CUDA_CONTROL(upload<int>(d_ws, ws, hlength));
    sobjs.ws = d_ws;
    //
    int *d_hs;
    CUDA_CONTROL(upload<int>(d_hs, hs, hlength));
    sobjs.hs = d_hs;
    //
    int *d_bpps;
    CUDA_CONTROL(upload<int>(d_bpps, bpps, hlength));
    sobjs.bpps = d_bpps;
    //
    int *d_tindices;
    CUDA_CONTROL(
        upload<int>(d_tindices, tindices, hlength));
    sobjs.tindices = d_tindices;
    //
    float *d_scales;
    CUDA_CONTROL(upload<float>(d_scales, scales, hlength));
    sobjs.scales = d_scales;
    return sobjs;
  }
  __host__ __device__ HittableParams
  get_hittable_params(int obj_index) {

    GroupType gtype =
        static_cast<GroupType>(gtypes[obj_index]);

    HittableType htype =
        static_cast<HittableType>(htypes[obj_index]);
    HittableParams hp(
        gtype, group_ids[obj_index], htype, rads[obj_index],
        Point3(p1xs[obj_index], p1ys[obj_index],
               p1zs[obj_index]),
        Point3(p2xs[obj_index], p2ys[obj_index],
               p2zs[obj_index]),
        Point3(p3xs[obj_index], p3ys[obj_index],
               p3zs[obj_index]));
    return hp;
  }
  __host__ __device__ ImageParams
  get_img_params(int obj_index) {
    ImageParams imp(ws[obj_index], hs[obj_index],
                    bpps[obj_index], tindices[obj_index]);
    return imp;
  }
  __host__ __device__ MatTextureParams
  get_mat_params(int obj_index) {
    //
    MaterialType mtype =
        static_cast<MaterialType>(mtypes[obj_index]);
    TextureType ttype =
        static_cast<TextureType>(ttypes[obj_index]);

    unsigned char *imdata;
    int imsize = 0;
    if (ttype == IMAGE) {
      imdata = tdata;
      imsize = tdata_size;
    }

    ImageParams imp = get_img_params(obj_index);

    MatTextureParams mp(
        mtype, fuzzs[obj_index], ttype,
        Color(tp1xs[obj_index], tp1ys[obj_index],
              tp1zs[obj_index]),
        Color(tp2xs[obj_index], tp2ys[obj_index],
              tp2zs[obj_index]),
        imdata, imsize, imp, scales[obj_index]);
    return mp;
  }
  __host__ __device__ ScenePrimitive get(int obj_index) {
    HittableParams hp = get_hittable_params(obj_index);
    MatTextureParams mp = get_mat_params(obj_index);

    ScenePrimitive sobj(hp, mp);
    return sobj;
  }
  __device__ ScenePrimitive get(int obj_index,
                                curandState *loc) {
    HittableParams hp = get_hittable_params(obj_index);
    MatTextureParams mp = get_mat_params(obj_index);
    ScenePrimitive sobj(hp, mp, loc);
    return sobj;
  }
  __host__ __device__ SceneGroup
  get_group(int group_index) {
    int group_start = group_starts[group_index];
    int group_size = group_sizes[group_index];
    GroupType gtype =
        static_cast<GroupType>(gtypes[group_index]);
    int group_id = group_ids[group_index];
    ScenePrimitive *prims = new ScenePrimitive[group_size];
    int gcount = 0;
    for (int i = group_start; i < group_size; i++) {
      ScenePrimitive prim = get(i);
      prims[gcount] = prim;
      gcount++;
    }
    SceneGroup sg(group_id, gcount, gtype, prims);
    return sg;
  }
  __device__ SceneGroup get_group(int group_index,
                                  curandState *loc) {
    int group_start = group_starts[group_index];
    int group_size = group_sizes[group_index];
    GroupType gtype =
        static_cast<GroupType>(gtypes[group_index]);
    int group_id = group_ids[group_index];
    ScenePrimitive *prims = new ScenePrimitive[group_size];
    int gcount = 0;
    for (int i = group_start; i < group_size; i++) {
      ScenePrimitive prim = get(i, loc);
      prims[gcount] = prim;
      gcount++;
    }
    SceneGroup sg(group_id, gcount, gtype, prims);
    return sg;
  }

  __host__ __device__ void
  to_hittable_list(Hittable **&hs) {
    hs = new Hittable *[nb_group];
    SceneGroup *groups = new SceneGroup[nb_group];
    for (int i = 0; i < nb_group; i++) {
      SceneGroup gr = get_group(i);
      groups[i] = gr;
    }
    order_scene(groups, nb_group);
    for (int i = 0; i < nb_group; i++) {
      hs[i] = groups[i].to_hittable_group();
    }
  }

  __device__ void to_hittable_list(Hittable **&hs,
                                   curandState *loc) {
    hs = new Hittable *[nb_group];
    SceneGroup *groups = new SceneGroup[nb_group];
    for (int i = 0; i < nb_group; i++) {
      SceneGroup gr = get_group(i, loc);
      groups[i] = gr;
    }
    order_scene(groups, nb_group);
    for (int i = 0; i < nb_group; i++) {
      hs[i] = groups[i].to_hittable_group();
    }
  }
};

#endif
