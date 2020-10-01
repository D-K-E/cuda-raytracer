// DBSCAN Acceleration Structure
#ifndef DBSCAN_CUH
#define DBSCAN_CUH

// k nearest neighbour acceleration structure
#include <nextweek/external.hpp>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

__host__ __device__ int neighbour_nb(Hittable **&hs,
                                     int gsize, int index,
                                     float min_limit) {
  //
  int counter = 0;
  Aabb temp;
  bool has_bbox = hs[index]->bounding_box(0, 0, temp);
  if (!has_bbox)
    return 0;
  Point3 bcenter = temp.center();
  for (int i = 0; i < gsize; i++) {
    Aabb t1;
    has_bbox = hs[i]->bounding_box(0, 1, t1);
    if (!has_bbox) {
      continue;
    }
    if (distance(bcenter, t1.center()) < min_limit) {
      counter++;
    }
  }
  return counter;
}

__host__ __device__ int *
get_neighbours(Hittable **&hs, int gsize, int index,
               float min_limit, int neighbour_nb) {
  int *indices = new int[neighbour_nb];
  int counter = 0;
  Aabb temp;
  bool has_bbox = hs[index]->bounding_box(0, 0, temp);
  if (!has_bbox)
    return 0;
  Point3 bcenter = temp.center();
  for (int dindex = 0; dindex < gsize; dindex++) {
    Aabb t1;
    has_bbox = hs[dindex]->bounding_box(0, 1, t1);
    if (!has_bbox) {
      continue;
    }
    if (distance(bcenter, t1.center()) < min_limit) {
      indices[counter] = dindex;
      counter++;
    }
  }
  return indices;
}

__host__ __device__ int *fill(int gsize, int v) {
  int *queue = new int[gsize];
  for (int i = 0; i < gsize; i++) {
    queue[i] = v;
  }
  return queue;
}

__host__ __device__ int count_positive(int *queue,
                                       int size) {
  int c = 0;
  for (int i = 0; i < size; i++) {
    if (queue[i] < 0)
      c++;
  }
  return c;
}

__host__ __device__ void
grow_set(Hittable **&hs, int *&labels, int gsize, int index,
         int label, float min_limit, int min_neighbour_nb) {
  int size = gsize;
  int *queue = fill(size, -1);
  int i = 0;
  queue[i] = index;
  int counter = 0;
  while (i < count_positive(queue, size)) {
    //
    index = queue[i];
    int nb_neigh =
        neighbour_nb(hs, gsize, index, min_limit);
    if (nb_neigh < min_neighbour_nb) {
      i++;
      continue;
    }
    int *neighbour_indices = get_neighbours(
        hs, gsize, index, min_limit, nb_neigh);
    for (int k = 0; k < nb_neigh; k++) {
      int neighbour_index = neighbour_indices[k];
      if (labels[neighbour_index] == -1) {
        labels[neighbour_index] = label;
      } else if (labels[neighbour_index] == 0) {
        labels[neighbour_index] = label;
        counter++;
        queue[counter] = neighbour_index;
      }
    }
    i++;
  }
}

__host__ __device__ void dbscan(Hittable **&hs, int gsize,
                                int *&labels,
                                float min_limit,
                                int min_neighbour_nb) {
  //
  labels = fill(gsize, 0);
  int label = 0;
  for (int index = 0; index < gsize; index++) {
    if (labels[index] != 0)
      continue;
    int nb_neigh =
        neighbour_nb(hs, gsize, index, min_limit);
    if (nb_neigh < min_neighbour_nb) {
      // noise
      labels[index] = -1;
    } else {
      label++;
      labels[index] = label;
      grow_set(hs, labels, gsize, index, label, min_limit,
               min_neighbour_nb);
    }
  }
}
#endif
