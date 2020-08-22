#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <oneweekend/vec3.hpp>
#include <oneweekend/color.hpp>
#include <oneweekend/ray.hpp>

void cuda_control(cudaError_t res, const char *const fn,
        const char * const f, const int l){
    if (res != cudaSuccess){
        std::cerr << "CUDA ERROR :: " 
            << static_cast<unsigned int>(res)
            << " "
            << cudaGetErrorName(res)
            << " file: " << f << " line: " << l 
            << " function: " << fn << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

#define CUDA_CONTROL(v) cuda_control((v), #v, __FILE__, __LINE__)

__device__ bool hit_sphere(const Point3& center, float radius, const Ray&r){
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4.0f*a*c;
    return (discriminant > 0.0f);
}

__device__ Color ray_color(const Ray & r){
    if (hit_sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, r)){
        return Vec3(1.0f, 0.0f, 0.5f);
    }
    Vec3 udir = to_unit(r.direction());
    float t = 0.5f * (udir.y() + 1.0f);
    return (1.0f-t)*Vec3(1.0f)+ t*Vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(Vec3 *fb, int maximum_x, int maximum_y,
        Vec3 lower_left, Vec3 horizontal, Vec3 vertical, Vec3 origin){
    int i = threadIdx.x + blockIdx.x  * blockDim.x;
    int j = threadIdx.y + blockIdx.y  * blockDim.y;

    if ((i >= maximum_x) || (j >= maximum_y)){
        return;
    }
    int pixel_index = j * maximum_x * 3 + i;
    float u = float(i) / float(maximum_x);
    float v = float(j) / float(maximum_y);
    Ray r(origin, lower_left + u * horizontal + v * vertical);
    fb[pixel_index] = ray_color(r);
}

int main(){
    int WIDTH = 1200;
    int HEIGHT = 600;
    int BLOCK_WIDTH = 8;
    int BLOCK_HEIGHT = 8;

    std::cerr << "Resim boyutumuz " << WIDTH << "x"
        << HEIGHT << std::endl;

    std::cerr << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << " bloklar halinde"
        << std::endl;

    
    // declare frame
    Vec3 *fb;

    // declare frame size
    int total_pixel_size = WIDTH * HEIGHT;
    size_t frameSize = 3 * total_pixel_size * sizeof(Vec3);

    CUDA_CONTROL(
            cudaMallocManaged(
                (void **)&fb, frameSize
                )
            );

    clock_t baslar, biter;
    baslar = clock();

    dim3 blocks(WIDTH / BLOCK_WIDTH + 1,
            HEIGHT / BLOCK_HEIGHT + 1);
    dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT); 
    render<<<blocks, threads>>>(fb, WIDTH, HEIGHT,
                                Vec3(-2.0f, -1.0f, -1.0f),
                                Vec3(4.0f, 0.0f, 0.0f),
                                Vec3(0.0f, 2.0f, 0.0f),
                                Vec3(0.0f));
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());
    biter = clock();
    double saniyeler = ((double)(biter - baslar)) / CLOCKS_PER_SEC;
    std::cerr << "Islem " << saniyeler << " saniye surdu" 
        << std::endl;

    std::cout << "P3" << std::endl;
    std::cout << WIDTH << " " << HEIGHT << std::endl;
    std::cout << "255" << std::endl;

    for (int j = HEIGHT - 1; j >= 0; j--){
        for (int i = 0; i < WIDTH; i++){
            size_t pixel_index = j*3*WIDTH + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " "
                << ib << std::endl;
        }
    }
    CUDA_CONTROL(cudaFree(fb));
}
