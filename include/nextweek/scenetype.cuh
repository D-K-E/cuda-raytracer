#ifndef SCENETYPES_CUH
#define SCENETYPES_CUH

enum TextureType : int {
  SOLID = 1,
  CHECKER = 2,
  NOISE = 3,
  IMAGE = 4
};

enum MaterialType : int {
  LAMBERT = 1,
  METAL = 2,
  DIELECTRIC = 3,
  DIFFUSE_LIGHT = 4,
  ISOTROPIC = 5
};

enum HittableType : int {
  LINE = 1,
  TRIANGLE = 2,
  SPHERE = 3,
  MOVING_SPHERE = 4,
  RECTANGLE = 5
};

enum GroupType : int { INSTANCE = 1, CONSTANT_MEDIUM = 2 };

#endif
