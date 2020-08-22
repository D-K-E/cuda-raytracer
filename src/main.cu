#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

void cuda_kontrol(cudaError_t sonuc, const char *const fn,
        const char * const dosya, const int satir){
    if (sonuc != cudaSuccess){
        std::cerr << "CUDA HATASI :: " 
            << static_cast<unsigned int>(sonuc)
            << " "
            << cudaGetErrorName(sonuc)
            << " dosya: " << dosya << " satir: " << satir 
            << " fonksiyon: " << fn << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

#define CUDA_HATA_KONTROL(deger) cuda_kontrol((deger), #deger, __FILE__, __LINE__)

__global__ void ciz(float *fb, int maximum_x, int maximum_y){
    int i = threadIdx.x + blockIdx.x  * blockDim.x;
    int j = threadIdx.y + blockIdx.y  * blockDim.y;

    if ((i >= maximum_x) || (j >= maximum_y)){
        return;
    }
    int piksel_indeksi = j * maximum_x * 3 + i * 3;
    fb[piksel_indeksi + 0] = float(i) / maximum_x;
    fb[piksel_indeksi + 1] = float(j) / maximum_y;
    fb[piksel_indeksi + 2] = 0.4f;
}

int main(){
    int genislik = 1200;
    int uzunluk = 600;
    int blok_genisligi = 8;
    int blok_uzunlugu = 8;

    std::cerr << "Resim boyutumuz " << genislik << "x"
        << uzunluk << std::endl;

    std::cerr << blok_genisligi << "x" << blok_uzunlugu << " bloklar halinde"
        << std::endl;

    int toplam_piksel_sayisi = genislik * uzunluk;
    size_t cerceveBoyutu = 3 * toplam_piksel_sayisi * sizeof(float);

    float *fb;
    CUDA_HATA_KONTROL(
            cudaMallocManaged(
                (void **)&fb, cerceveBoyutu
                )
            );

    clock_t baslar, biter;
    baslar = clock();

    dim3 bloklar(genislik / blok_genisligi + 1,
            uzunluk / blok_uzunlugu + 1);
    dim3 threadler(blok_genisligi, blok_uzunlugu); 
    ciz<<<bloklar, threadler>>>(fb, genislik, uzunluk);
    CUDA_HATA_KONTROL(cudaGetLastError());
    CUDA_HATA_KONTROL(cudaDeviceSynchronize());
    biter = clock();
    double saniyeler = ((double)(biter - baslar)) / CLOCKS_PER_SEC;
    std::cerr << "Islem " << saniyeler << " saniye surdu" 
        << std::endl;

    std::cout << "P3" << std::endl;
    std::cout << genislik << " " << uzunluk << std::endl;
    std::cout << "255" << std::endl;

    for (int j = uzunluk - 1; j >= 0; j--){
        for (int i = 0; i < genislik; i++){
            size_t piksel_indeksi = j*3*genislik + i * 3;
            float r = fb[piksel_indeksi + 0];
            float g = fb[piksel_indeksi + 1];
            float b = fb[piksel_indeksi + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " "
                << ib << std::endl;
        }
    }
    CUDA_HATA_KONTROL(cudaFree(fb));
}
