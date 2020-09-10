#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include <nextweek/external.hpp>
//
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

class Texture {
public:
  __device__ virtual Color value(float u, float v,
                                 const Point3 &p) const = 0;
};

class SolidColor : public Texture {
public:
  __device__ SolidColor() {}
  __device__ SolidColor(Color c) : color_value(c) {}

  __device__ SolidColor(float red, float green, float blue)
      : SolidColor(Color(red, green, blue)) {}

  __device__ Color value(float u, float v,
                         const Point3 &p) const override {
    return color_value;
  }

private:
  Color color_value;
};

class CheckerTexture : public Texture {
public:
  Texture *odd;
  Texture *even;

public:
  __device__ CheckerTexture() {}
  __device__ ~CheckerTexture() {
    delete odd;
    delete even;
  }
  __device__ CheckerTexture(Color c1, Color c2) {
    odd = new SolidColor(c1);
    even = new SolidColor(c2);
  }
  __device__ CheckerTexture(Color c1) {
    odd = new SolidColor(c1);
    even = new SolidColor(1.0 - c1);
  }

  __device__ CheckerTexture(Texture *c1, Texture *c2) {
    odd = c1;
    even = c2;
  }
  __device__ Color value(float u, float v,
                         const Point3 &p) const override {
    //
    float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                  sin(10.0f * p.z());
    if (sines < 0) {
      return odd->value(u, v, p);
    } else {
      return even->value(u, v, p);
    }
  }
};

struct ImageData {
  unsigned char *data;
  int width, height;
  int bytes_per_line, bytes_per_pixel;
  int nb_image;
  int size;
  __host__ __device__ ImageData()
      : data(nullptr), width(0), height(0),
        bytes_per_line(0), bytes_per_pixel(0), nb_image(0),
        size(0) {}
  __device__ ImageData(unsigned char *imdata, int w, int h,
                       int bpl, int bpp, int nbim) {
    data = imdata;
    width = w;
    height = h;
    bytes_per_pixel = bpp;
    bytes_per_line = bpl;
    nb_image = nbim;
    size = bytes_per_line * height * nb_image;
  }
  __host__ __device__ ~ImageData() { delete data; }
  __host__ ImageData(const char *impath) {
    nb_image = 1;
    imread(impath);
    size = height * bytes_per_line;
  }
  __host__ ImageData(std::vector<const char *> impaths) {
    nb_image = impaths.size();
    imread(impaths);
    size = height * bytes_per_line * nb_image;
  }
  __host__ __device__ void
  get_image_data(int index, int size,
                 unsigned char *imdata) const {
    int start_index = index * size;
    for (int i = 0; i < size; i++) {
      imdata[i] = data[i + start_index];
    }
  }

  __host__ void set_im_vals(int w, int h, int per_pixel) {
    width = w;
    height = h;
    bytes_per_pixel = per_pixel;
    bytes_per_line = width * bytes_per_pixel;
  }
  __host__ void imread(const char *impath) {
    int w, h, comp;
    data = stbi_load(impath, &w, &h, &comp, 0);
    if (!data) {
      throw std::runtime_error("Image can not be loaded");
    }
    set_im_vals(w, h, comp);
  }
  __host__ void imread(std::vector<const char *> impaths) {
    std::vector<unsigned char *> ims(impaths.size());
    for (int i = 0; i < impaths.size(); i++) {
      int w, h, comp;
      ims[i] = stbi_load(impaths[i], &w, &h, &comp, 0);
    }
    unsigned char
        imdata[height * bytes_per_line * nb_image];
    for (int i = 0; i < ims.size(); i++) {
      unsigned char *imd = ims[i];
      for (int k = 0; k < height * bytes_per_line; k++) {
        imdata[i * k] = imd[k];
      }
    }
    data = imdata;
  }
};

class ImageTexture : public Texture {
public:
  unsigned char *data;
  int width, height;
  int bytes_per_line; // == bytes_per_pixel * width;
  int bytes_per_pixel;

public:
  __host__ __device__ ImageTexture()
      : data(nullptr), width(0), height(0),
        bytes_per_line(0), bytes_per_pixel(0) {}
  __host__ ~ImageTexture() { cudaFree(data); }
  __host__ ImageTexture(const char *impath) {
    imread(impath);
  }
  __host__ void imread(const char *impath) {
    int w, h, comp;
    data = stbi_load(impath, &w, &h, &comp, 0);
    if (!data) {
      throw std::runtime_error("Image can not be loaded");
    }
    set_im_vals(w, h, comp);
  }
  __host__ void set_im_vals(int w, int h, int per_pixel) {
    width = w;
    height = h;
    bytes_per_pixel = per_pixel;
    bytes_per_line = width * bytes_per_pixel;
  }
  __device__ ImageTexture(unsigned char *d, int w, int h,
                          int bpl, int bpp)
      : data(d), width(w), height(h), bytes_per_line(bpl),
        bytes_per_pixel(bpp) {}
  __device__ Color value(float u, float v,
                         const Point3 &p) const override {
    if (data == nullptr) {
      return Color(1.0, 0.0, 0.0);
    }
    u = clamp(u, 0.0, 1.0);
    v = 1.0 - clamp(v, 0.0, 1.0); // flip v to im coords
    int xi = (int)(u * width);
    int yj = (int)(v * height);
    xi = xi >= width ? width - 1 : xi;
    yj = yj >= height ? height - 1 : yj;

    //
    int pixel = yj * bytes_per_line + xi * bytes_per_pixel;
    float r = (float)data[pixel] / 255;
    float g = (float)data[pixel + 1] / 255;
    float b = (float)data[pixel + 2] / 255;
    return Color(r, g, b);
  }
};

#endif
