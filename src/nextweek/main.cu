// libs
#include <nextweek/camera.cuh>
#include <nextweek/cbuffer.hpp>
#include <nextweek/color.hpp>
#include <nextweek/debug.hpp>
#include <nextweek/external.hpp>
#include <nextweek/hittables.cuh>
#include <nextweek/kernels/makeworld.cuh>
#include <nextweek/kernels/trace.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/texture.cuh>
#include <nextweek/vec3.cuh>

__global__ void rand_init(curandState *randState,
                          int seed) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(seed, 0, 0, randState);
  }
}

__global__ void render_init(int mx, int my,
                            curandState *randState,
                            int seed) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    curand_init(seed, 0, 0, randState);
  }
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= mx) || (j >= my)) {
    return;
  }
  int pixel_index = j * mx + i;
  // same seed, different index
  curand_init(seed + pixel_index, pixel_index, 0,
              &randState[pixel_index]);
}

void get_device_props() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cerr << "Device Number: " << i << std::endl;
    std::cerr << "Device name: " << prop.name << std::endl;
    std::cerr << "Memory Clock Rate (KHz): "
              << prop.memoryClockRate << std::endl;
    std::cerr << "Memory Bus Width (bits): "
              << prop.memoryBusWidth << std::endl;
    std::cerr << "  Peak Memory Bandwidth (GB/s): "
              << 2.0 * prop.memoryClockRate *
                     (prop.memoryBusWidth / 8) / 1.0e6
              << std::endl;
  }
}

int main() {
  float aspect_ratio = 16.0f / 9.0f;
  int WIDTH = 320;
  int HEIGHT = static_cast<int>(WIDTH / aspect_ratio);
  int BLOCK_WIDTH = 32;
  int BLOCK_HEIGHT = 18;
  int SAMPLE_NB = 50;
  int BOUNCE_NB = 50;

  get_device_props();

  std::cerr << "Resim boyutumuz " << WIDTH << "x" << HEIGHT
            << std::endl;

  std::cerr << BLOCK_WIDTH << "x" << BLOCK_HEIGHT
            << " bloklar halinde" << std::endl;

  // declare frame size
  int total_pixel_size = WIDTH * HEIGHT;
  size_t frameSize = 3 * total_pixel_size;

  // declare frame
  thrust::device_ptr<Vec3> fb =
      thrust::device_malloc<Vec3>(frameSize);
  CUDA_CONTROL(cudaGetLastError());

  // declare random state
  int SEED = time(NULL);
  thrust::device_ptr<curandState> randState1 =
      thrust::device_malloc<curandState>(frameSize);
  CUDA_CONTROL(cudaGetLastError());

  // declare random state 2
  thrust::device_ptr<curandState> randState2 =
      thrust::device_malloc<curandState>(1);
  CUDA_CONTROL(cudaGetLastError());
  rand_init<<<1, 1>>>(thrust::raw_pointer_cast(randState2),
                      SEED);
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  // declare world
  thrust::device_ptr<Hittables *> world =
      thrust::device_malloc<Hittables *>(1);
  CUDA_CONTROL(cudaGetLastError());
  int box_size = 6;
  int side_box_nb = 15;
  int sphere_nb = 4;
  int nb_hittable = side_box_nb;
  nb_hittable *= side_box_nb;
  nb_hittable *= box_size;
  nb_hittable += sphere_nb;
  // nb_hittable += 1;
  thrust::device_ptr<Hittable *> hs =
      thrust::device_malloc<Hittable *>(nb_hittable);
  CUDA_CONTROL(cudaGetLastError());

  // declara imdata
  // std::vector<const char *> impaths =
  // {"media/earthmap.png",
  //                                     "media/lsjimg.png"};
  // std::vector<int> ws, hes, nbChannels;
  // int totalSize;
  // std::vector<unsigned char> imdata_h;
  // imread(impaths, ws, hes, nbChannels, imdata_h,
  // totalSize);
  ////// thrust::device_ptr<unsigned char> imda =
  //////    thrust::device_malloc<unsigned char>(imd.size);
  // unsigned char *h_ptr = imdata_h.data();

  ////// --------------------- image
  ///------------------------
  // thrust::device_ptr<unsigned char> imdata;
  // upload_to_device(imdata, h_ptr, imdata_h.size());

  // int *ws_ptr = ws.data();

  // thrust::device_ptr<int> imwidths;
  // upload_to_device(imwidths, ws_ptr, ws.size());

  // thrust::device_ptr<int> imhs;
  // int *hs_ptr = hes.data();
  // upload_to_device(imhs, hs_ptr, hes.size());

  // thrust::device_ptr<int> imch; // nb channels
  // int *nb_ptr = nbChannels.data();
  // upload_to_device(imch, nb_ptr, nbChannels.size());

  // CUDA_CONTROL(cudaGetLastError());

  make_world<<<1, 1>>>(
      thrust::raw_pointer_cast(world),
      thrust::raw_pointer_cast(hs),
      thrust::raw_pointer_cast(randState2), side_box_nb
      // thrust::raw_pointer_cast(imdata),
      // thrust::raw_pointer_cast(imwidths),
      // thrust::raw_pointer_cast(imhs),
      // thrust::raw_pointer_cast(imch)
      );
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  clock_t baslar, biter;
  baslar = clock();

  dim3 blocks(WIDTH / BLOCK_WIDTH + 1,
              HEIGHT / BLOCK_HEIGHT + 1);
  dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
  render_init<<<blocks, threads>>>(
      WIDTH, HEIGHT, thrust::raw_pointer_cast(randState1),
      SEED + 7);
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());

  // declare camera

  // one weekend final camera specification
  // Vec3 lookfrom(13, 2, 3);
  // Vec3 lookat(0, 0, 0);
  // Vec3 wup(0, 1, 0);
  // float vfov = 20.0f;
  // float aspect_r = float(WIDTH) / float(HEIGHT);
  // float dist_to_focus = 10.0;
  //(lookfrom - lookat).length();
  // float aperture = 0.1;
  // float t0 = 0.0f, t1 = 1.0f;

  // nextweek empty cornell box specification

  Vec3 lookfrom(478, 278, -600);
  Vec3 lookat(278, 278, 0);
  Vec3 wup(0, 1, 0);
  float vfov = 40.0f;
  float aspect_r = float(WIDTH) / float(HEIGHT);
  float dist_to_focus = (lookfrom - lookat).length();
  float aperture = 0.0;
  float t0 = 0.0f, t1 = 1.0f;

  Camera cam(lookfrom, lookat, wup, vfov, aspect_r,
             aperture, dist_to_focus, t0, t1);

  render<<<blocks, threads>>>(
      thrust::raw_pointer_cast(fb), WIDTH, HEIGHT,
      SAMPLE_NB, BOUNCE_NB, cam,
      thrust::raw_pointer_cast(world),
      thrust::raw_pointer_cast(randState1));
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaDeviceSynchronize());
  biter = clock();
  double saniyeler =
      ((double)(biter - baslar)) / CLOCKS_PER_SEC;
  std::cerr << "Islem " << saniyeler << " saniye surdu"
            << std::endl;

  std::cout << "P3" << std::endl;
  std::cout << WIDTH << " " << HEIGHT << std::endl;
  std::cout << "255" << std::endl;

  for (int j = HEIGHT - 1; j >= 0; j--) {
    for (int i = 0; i < WIDTH; i++) {
      size_t pixel_index = j * WIDTH + i;
      thrust::device_reference<Vec3> pix_ref =
          fb[pixel_index];
      Vec3 pixel = pix_ref;
      int ir = int(255.99 * pixel.r());
      int ig = int(255.99 * pixel.g());
      int ib = int(255.99 * pixel.b());
      std::cout << ir << " " << ig << " " << ib
                << std::endl;
    }
  }
  CUDA_CONTROL(cudaDeviceSynchronize());
  CUDA_CONTROL(cudaGetLastError());
  free_world(fb,    //
             world, //
             hs,    //
             // imdata, imch, imhs, imwidths, //
             randState1, //
             randState2);
  // free_world(fb, world, hs, randState1, randState2);
  CUDA_CONTROL(cudaGetLastError());

  cudaDeviceReset();
}
