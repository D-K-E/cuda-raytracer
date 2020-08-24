#include <oneweekend/vec3.cuh>
#include <oneweekend/color.hpp>
#include <oneweekend/ray.cuh>
#include <oneweekend/sphere.cuh>
#include <oneweekend/hittables.cuh>
#include <oneweekend/camera.cuh>
#include <oneweekend/material.cuh>
#include <oneweekend/debug.hpp>
#include <oneweekend/external.hpp>


__device__ Color ray_color(
        const Ray & r,
        Hittables** world, 
        curandState *local_rand_state,
        int bounceNb){
    Ray current_ray = r;
    Vec3 current_attenuation = Vec3(1.0f);
    while (bounceNb > 0){
        HitRecord rec;
        bool anyHit = world[0]->hit(current_ray, 0.001f, FLT_MAX, rec);
        if (anyHit){
            Ray scattered;
            Vec3 attenuation;
            bool isScattered = rec.mat_ptr->scatter(
                    current_ray, 
                    rec,
                    attenuation, 
                    scattered, 
                    local_rand_state);
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

__global__ void rand_init(curandState * randState,
        int seed){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        curand_init(seed, 0,0, randState);
    }
}

__global__ void render_init(int mx, int my, curandState *randState, int seed){
    if (threadIdx.x == 0 && threadIdx.y == 0){
        curand_init(seed, 0,0, randState);
    }
    int i = threadIdx.x + blockIdx.x  * blockDim.x;
    int j = threadIdx.y + blockIdx.y  * blockDim.y;

    if ((i >= mx) || (j >= my)){
        return;
    }
    int pixel_index = j * mx + i;
    // same seed, different index
    curand_init(seed, pixel_index, 0, &randState[pixel_index]);
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
    int pixel_index = j * maximum_x + i;
    curandState localS = randState[pixel_index];
    Vec3 rcolor(0.0f);
    for(int s = 0; s < sample_nb; s++){
        float u = float(i + curand_uniform(&localS)) / float(maximum_x);
        float v = float(j+ curand_uniform(&localS)) / float(maximum_y);
        Ray r = cam[0]->get_ray(u,v, &localS);
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
        Camera** cam, int nx, int ny, curandState * randState, int row){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        // declare objects
        curandState local_rand_state = *randState;
        Lambertian* lamb = new Lambertian(Vec3(0.5, 0.5, 0.5));
        ss[0] = new Sphere(Vec3(0,-1000.0,-1), 1000, lamb);
        int i = 1;
        int halfRow = row / 2;
        for(int a = -halfRow; a < halfRow; a++) {
            for(int b = -halfRow; b < halfRow; b++) {
                float choose_mat = RND;
                Vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    Material* lamb1 = new Lambertian(Vec3(RND*RND, RND*RND,
                                RND*RND));
                    ss[i++] = new Sphere(center, 0.2, lamb1);
                }
                else if(choose_mat < 0.95f) {
                    Material * met = new Metal(
                            Vec3(
                                0.5f*(1.0f+RND),
                                0.5f*(1.0f+RND), 
                                0.5f*(1.0f+RND)
                                ), 0.5f*RND);
                    ss[i++] = new Sphere(center, 0.2, met);
                }
                else {
                    Material * diel = new Dielectric(1.5);
                    ss[i++] = new Sphere(center, 0.2, diel);
                }
            }
        }

        Material* diel = new Dielectric(1.5);
        ss[i++] = new Sphere(Vec3(0, 1,0),  1.0, diel);

        Material* lamb2 = new Lambertian(Vec3(0.4, 0.2, 0.1));
        ss[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, lamb2);

        Material* met2 = new Metal(Vec3(0.7, 0.6, 0.5), 0.0);
        ss[i++] = new Sphere(Vec3(4, 1, 0),  1.0, met2);
        randState[0] = local_rand_state;
        world[0]  = new Hittables(ss, 22*22+1+3);

        Vec3 lookfrom(13,2,3);
        Vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        cam[0] = new Camera(lookfrom,
                lookat,
                Vec3(0,1,0),
                30.0,
                float(nx)/float(ny),
                aperture,
                dist_to_focus);
    }
}
__global__ void free_world(Hittables** world,Hittable **ss, Camera**cam){
    int size=22*22+1+3;
    for(int i=0; i < size; i++) {
        delete ((Sphere *)ss[i])->mat_ptr;
        delete ss[i];
    }
    delete world[0];
    delete cam[0];
}

int main(){
    int WIDTH = 800;
    int HEIGHT = 600;
    int BLOCK_WIDTH = 8;
    int BLOCK_HEIGHT = 8;
    int SAMPLE_NB = 50;
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
    int SEED = 1987;
    thrust::device_ptr<curandState> randState1 = thrust::device_malloc<curandState>(frameSize);
    CUDA_CONTROL(cudaGetLastError());

    // declare random state 2
    thrust::device_ptr<curandState> randState2 = thrust::device_malloc<curandState>(1);
    CUDA_CONTROL(cudaGetLastError());
    rand_init<<<1,1>>>(
            thrust::raw_pointer_cast(randState2),
            SEED);
    CUDA_CONTROL(cudaGetLastError());
    CUDA_CONTROL(cudaDeviceSynchronize());


    // declare world
    thrust::device_ptr<Hittables*> world = thrust::device_malloc<Hittables*>(1);
    CUDA_CONTROL(cudaGetLastError());
    int row = 22;
    int focus_obj_nb = 3;
    int nb_hittable = row * row +1 +focus_obj_nb;
    thrust::device_ptr<Hittable*> hs = thrust::device_malloc<Hittable*>(nb_hittable);
    CUDA_CONTROL(cudaGetLastError());

    // declare camera
    thrust::device_ptr<Camera*> cam = thrust::device_malloc<Camera*>(1);
    CUDA_CONTROL(cudaGetLastError());

    make_world<<<1,1>>>(
            thrust::raw_pointer_cast(world),
            thrust::raw_pointer_cast(hs),
            2,
            thrust::raw_pointer_cast(cam),
            WIDTH, HEIGHT, 
            thrust::raw_pointer_cast(randState2),
            row
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
            thrust::raw_pointer_cast(randState1),
            SEED
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
            thrust::raw_pointer_cast(randState1)
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
            size_t pixel_index = j*WIDTH + i;
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
    thrust::device_free(randState2);
    CUDA_CONTROL(cudaGetLastError());
    thrust::device_free(randState1);
    CUDA_CONTROL(cudaGetLastError());
    cudaDeviceReset();
}
