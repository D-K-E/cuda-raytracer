// DBSCAN Acceleration Structure
#ifndef DBSCAN_CUH
#define DBSCAN_CUH

// k nearest neighbour acceleration structure
#include <nextweek/external.hpp>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

struct HLabel {
  int label;
  Hittable *obj;
  int index;
  __host__ __device__ HLabel() {}
  __host__ __device__ HLabel(int lbl, Hittable *&ob,
                             int inx)
      : label(lbl), obj(ob), index(inx) {}
};
__host__ __device__ bool operator>(const HLabel &hl1,
                                   const HLabel &hl2) {
  return hl1.label > hl2.label;
}
__host__ __device__ bool operator>(HLabel *hl1,
                                   HLabel *hl2) {
  return hl1->label > hl2->label;
}

struct HittableLabels {
  int *labels;
  Hittable **groups;
  const unsigned int group_size;
  __host__ __device__ HittableLabels()
      : group_size(0), labels(nullptr), groups(nullptr) {}

  __host__ __device__ HittableLabels(unsigned int gsize)
      : group_size(gsize), labels(new int[group_size]),
        groups(new Hittable *[group_size]) {}
  __host__ __device__ HittableLabels(int gsize)
      : HittableLabels(static_cast<unsigned int>(gsize)) {}
  __host__ __device__ HittableLabels(int *lbls,
                                     unsigned int gsize,
                                     Hittable **&gs)
      : labels(lbls), group_size(gsize), groups(gs) {}
  __host__ __device__ HittableLabels(HLabel *&labels,
                                     unsigned int lblsize)
      : group_size(lblsize), labels(new int[group_size]),
        groups(new Hittable *[group_size]) {
    for (unsigned int i = 0; i < group_size; i++) {
      insert(labels[i]);
    }
  }
  __host__ __device__ bool insert(int label, Hittable *obj,
                                  int index) {
    bool result = index < group_size && index > 0;
    if (result) {
      groups[index] = obj;
      labels[index] = label;
    }
    return result;
  }
  __host__ __device__ bool insert(HLabel &hlbl) {
    return insert(hlbl.label, hlbl.obj, hlbl.index);
  }
  __host__ __device__ bool get(int index, HLabel &hlbl) {
    bool result = index < group_size && index > 0;
    if (result) {
      hlbl = HLabel(labels[index], groups[index], index);
    }
    return result;
  }
  __host__ __device__ void to_hlabel_array(HLabel *&harr) {
    for (unsigned int i = 0; i < group_size; i++) {
      harr[i] = HLabel(labels[i], groups[i], i);
    }
  }
};

class DBSCAN {
public:
  HittableLabels labels;
  const float threshold;
  const int min_neighbor_nb;

public:
  __host__ __device__ DBSCAN()
      : threshold(0.0f), min_neighbor_nb(0) {}

  __host__ __device__ DBSCAN(Hittable **&gs,
                             unsigned int gsize,
                             float thresh, int mnb)
      : threshold(thresh), min_neighbor_nb(mnb),
        labels(gsize) {
    for (unsigned int i = 0; i < labels.group_size; i++) {
      HLabel hl(0, gs[i], i);
      labels.insert(hl);
    }
    int label = 0;
    for (unsigned int index = 0; index < labels.group_size;
         index++) {
      HLabel hl;
      labels.get(index, hl);
      if (hl.label != 0) {
        continue;
      }
      int nb_neighbour = neighbour_number(index);
      if (nb_neighbour < min_neighbor_nb) {
        // noise
        hl.label = -1;
        labels.insert(hl);
      } else {
        label++;
        hl.label = label;
        labels.insert(hl);
        grow_set(index, label);
      }
    }
  }

  __host__ __device__ Point3 bbox_center(int index) {
    Hittable *h = labels.groups[index];
    Aabb temp;
    if (h->bounding_box(0, 1, temp)) {
      return temp.center();
    } else {
      return Point3(FLT_MAX, FLT_MAX, FLT_MAX);
    }
  }
  __host__ __device__ int neighbour_number(int obj_index) {
    Point3 hcenter = bbox_center(obj_index);
    int counter = 0;
    for (int i = 0; i < obj_index; i++) {
      Point3 nc = bbox_center(i);
      if (distance(hcenter, nc) < threshold) {
        counter++;
      }
    }
    for (int i = obj_index + 1; i < group_size; i++) {
      Point3 nc = bbox_center(i);
      if (distance(hcenter, nc) < min_distance) {
        counter++;
      }
    }
    return counter;
  }
  __host__ __device__ int
  count_positive_elements(int *&arr) {
    int counter = 0;
    for (int i = 0; i < group_size; i++) {
      if (arr[i] >= 0) {
        counter++;
      }
    }
    return counter;
  }
  __host__ __device__ void fill_neg(int *&queue) {
    for (int i = 0; i < group_size; i++) {
      queue[i] = -2;
    }
  }
  __host__ __device__ void
  get_neighbour_index(int obj_index, int *&neighbours) {
    Point3 hcenter = bbox_center(obj_index);
    int counter = 0;
    for (int i = 0; i < obj_index; i++) {
      Point3 nc = bbox_center(i);
      if (distance(hcenter, nc) < threshold) {
        neighbours[counter] = i;
        counter++;
      }
    }
    for (int i = obj_index + 1; i < group_size; i++) {
      Point3 nc = bbox_center(i);
      if (distance(hcenter, nc) < min_distance) {
        neighbours[counter] = i;
        counter++;
      }
    }
  }
  __host__ __device__ void grow_set(int &index,
                                    int &label) {
    int *search_queue = new int[group_size];
    fill_neg(search_queue);
    int queue_counter = 0;
    search_queue[queue_counter] = index;
    int i = 0;
    while (i < count_positive_elements(search_queue)) {
      index = search_queue[i];
      int nb_neighbour = neighbour_number(index);
      if (nb_neighbour < min_neighbor_nb) {
        i++;
        continue;
      }
      int *neighbour_indices = new int[nb_neighbour];
      get_neighbour_index(obj_index, neighbour_indices);
      for (int k = 0; k < nb_neighbour; k++) {
        int neighbour_index = neighbour_indices[k];
        HLabel hlabel;

        if (labels.get(neighbour_index, hlabel)) {
          if (hlabel.label == -1) {
            hlabel.label = label;
          } else if (hlabel.label == 0) {
            hlabel.label = label;
            queue_counter++;
            search_queue[queue_counter] = neighbour_index;
          }
        }
      }
      i++;
    }
  }
  __host__ __device__ int get_nb_labels() {
    sorted_labels();
    int counter = 0;
    for (int i = 0; i < labels.group_size; i++) {
      while (i < labels.group_size - 1 &&
             labels.labels[i] == labels.labels[i + 1]) {
        i++;
      }
      counter++;
    }
    return counter;
  }
  __host__ __device__ int get_label_size(int label) {
    int c = 0;
    for (int i = 0; i < labels.group_size; i++) {
      if (labels.labels[i] == label) {
        c++;
      }
    }
    return c;
  }
  __host__ __device__ void sorted_labels() {
    odd_even_sort(labels.labels, labels.groups,
                  labels.group_size);
  }
};

#endif
