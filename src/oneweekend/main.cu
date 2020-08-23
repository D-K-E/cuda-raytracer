#include <oneweekend/vec3.cuh>
#include <oneweekend/color.hpp>
#include <oneweekend/ray.cuh>
#include <oneweekend/sphere.cuh>
#include <oneweekend/hittables.cuh>
#include <oneweekend/camera.cuh>
#include <oneweekend/debug.hpp>
#include <oneweekend/external.hpp>


__device__ Color ray_color(
        const Ray & r,
        Hittables** world, 
        curandState *local_rand_state,
        int bounceNb){
    Ray current_ray = r;
    float current_attenuation = 1.0f;
    while (bounceNb > 0){
        HitRecord rec;
        bool anyHit = world[0]->hit(current_ray, 0.001f, FLT_MAX, rec);
        if (anyHit){
            Ray scattered;
            Vec3 attenuation;
            bool isScattered = rec.mat_ptr->scatter(current_ray, 
                    attenuation, scattered, local_rand_state);
            if (isScattered){
                bounceNb--;
                current_attenuation *= attenuation;
                current_ray = scattered;
            }else{
                return Vec3(0.0f); // background color
            }
        } else {
            Vec3 udir = to_unit(current_ray.direction());
            float t = 0.5f * (udir.y() + 1.0f);
            Vec3 out = (1.0f-t)*Vec3(1.0f)+ t*Vec3(0.5f, 0.7f, 1.0f);
            return current_attenuation * out;
        }
    }
    return Vec3(0.0f); // background color

}

__global__ void render_init(curandState *randState){
    if (threadIdx.x == 0 && threadIdx.y == 0){
        curand_int(1923, 0,0, randState);
    }
    int i = threadIdx.x + blockIdx.x  * blockDim.x;
    int j = threadIdx.y + blockIdx.y  * blockDim.y;

    if ((i >= mx) || (j >= my)){
        return;
    }
    int pixel_index = j * mx * 3 + i;
    // same seed, different index
    curand_init(1987, pixel_index, 0, &randState[pixel_index]);
}

__global__ void render(
        Vec3 *fb, int maximum_x, int maximum_y, int sample_nb, int bounceNb,
        Camera** cam,
        Hittables** world,
        curandState *randState){
    int i = threadIdx.x + blockIdx.x  * blockDim.x;
    int j = threadIdx.y + blockIdx.y  * blockDim.y;

    if ((i >= maximum_x) || (j >= maximum_y)){
        return;
    }
    int pixel_index = j * maximum_x * 3 + i;
    curandState localS = randState[pixel_index];
    Vec3 rcolor(0.0f);
    for(int s = 0; s < sample_nb; s++){
        float u = float(i + curand_uniform(&localS)) / float(maximum_x);
        float v = float(j+ curand_uniform(&localS)) / float(maximum_y);
        Ray r = cam[0]->get_ray(u,v);
        rcolor += ray_color(r, world, randState, bounceNb);
    }
    // fix the bounce depth
    randState[pixel_index] = localS;
    rcolor /= float(sample_nb);
    rcolor.e[0] = sqrt(rcolor.x());
    rcolor.e[1] = sqrt(rcolor.y());
    rcolor.e[2] = sqrt(rcolor.z());
    fb[pixel_index] = rcolor;
}


__global__ void make_world(Hittables** world, Hittable** ss, int size,
        Camera** cam, int nx, int ny, curandState * randState){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        // declare objects
        curandState local_rand_state = *rand_state;
        ss[0] = new Sphere(vec3(0,-1000.0,-1), 1000,
                new Lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                Vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    ss[i++] = new Sphere(center, 0.2,
                            new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    ss[i++] = new Sphere(center, 0.2,
                            new Metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    ss[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
                }
            }
        }
        ss[i++] = new Sphere(Vec3(0, 1,0),  1.0, new Dielectric(1.5));
        ss[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(vec3(0.4, 0.2, 0.1)));
        ss[i++] = new Sphere(Vec3(4, 1, 0),  1.0, new Metal(vec3(0.7, 0.6, 0.5), 0.0));
        rand_state[0] = local_rand_state;
        world[0]  = new Hittables(d_list, 22*22+1+3);

        Vec3 lookfrom(13,2,3);
        Vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        cam[0]   = new camera(lookfrom,
                lookat,
                Vec3(0,1,0),
                30.0,
                float(nx)/float(ny),
                aperture,
                dist_to_focus);
    }
}
__global__ void free_world(Hittables** world,Hittable **ss, Camera**cam, int size=22*22+1+3){
    for(int i=0; i < size; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete ss[i];
    }
    delete ss[0];
    delete ss[1];
    delete world[0];
    delete cam[0];
}

int main(){
    int WIDTH = 1200;
    int HEIGHT = 600;
    int BLOCK_WIDTH = 8;
    int BLOCK_HEIGHT = 8;
    int SAMPLE_NB = 120;
    int BOUNCE_NB = 50;

    std::cerr << "Resim boyutumuz " << WIDTH << "x"
        << HEIGHT << std::endl;

    std::cerr << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << " bloklar halinde"
        << std::endl;


    // declare frame size
    int total_pixel_size = WIDTH * HEIGHT;
    size_t frameSize = 3 * total_pixel_size;

    // declare frame
    thrust::device_ptr<Vec3> fb = thrust::device_malloc<Vec3>(frameSize);
    CUDA_CONTROL(cudaGetLastError());

    // declare random state
    thrust::device_ptr<curandState> randState = thrust::device_malloc<curandState>(frameSize);
    CUDA_CONTROL(cudaGetLastError());

    // declare world
    thrust::device_ptr<Hittables*> world = thrust::device_malloc<Hittables*>(1);
    CUDA_CONTROL(cudaGetLastError());
    thrust::device_ptr<Hittable*> hs = thrust::device_malloc<Hittable*>(2);
    CUDA_CONTROL(cudaGetLastError());

    // declare camera
    thrust::device_ptr<Camera*> cam = thrust::device_malloc<Camera*>(1);
    CUDA_CONTROL(cudaGetLastError());

    make_world<<<1,1>>>(
            thrust::raw_pointer_cast(world),
            thrust::raw_pointer_cast(hs),
            2,
            thrust::raw_pointer_cast(cam)
            );
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    clock_t baslar, biter;
    baslar = clock();

    dim3 blocks(WIDTH / BLOCK_WIDTH + 1,
            HEIGHT / BLOCK_HEIGHT + 1);
    dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT); 
    render_init<<<blocks, threads>>>(
            WIDTH, 
            HEIGHT,
            thrust::raw_pointer_cast(randState)
            );
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(
            thrust::raw_pointer_cast(fb), 
            WIDTH, 
            HEIGHT,
            SAMPLE_NB,
            BOUNCE_NB,
            thrust::raw_pointer_cast(cam),
            thrust::raw_pointer_cast(world),
            thrust::raw_pointer_cast(randState)
            );
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
            thrust::device_reference<Vec3> pix_ref = fb[pixel_index];
            Vec3 pixel = pix_ref;
            int ir = int(255.99 * pixel.r());
            int ig = int(255.99 * pixel.g());
            int ib = int(255.99 * pixel.b());
            std::cout << ir << " " << ig << " "
                << ib << std::endl;
        }
    }
    CUDA_CONTROL(cudaDeviceSynchronize());
    free_world<<<1,1>>>(
            thrust::raw_pointer_cast(world),
            thrust::raw_pointer_cast(hs),
            thrust::raw_pointer_cast(cam)
            );
    CUDA_CONTROL(cudaGetLastError());
    thrust::device_free(fb);
    CUDA_CONTROL(cudaGetLastError());
    thrust::device_free(world);
    CUDA_CONTROL(cudaGetLastError());
    thrust::device_free(hs);
    CUDA_CONTROL(cudaGetLastError());
    thrust::device_free(cam);
    CUDA_CONTROL(cudaGetLastError());
    thrust::device_free(randState);
    CUDA_CONTROL(cudaGetLastError());
    cudaDeviceReset();
}
