#ifndef BVH_CUH
#define BVH_CUH

#include <nextweek/external.hpp>
#include <nextweek/hittable.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

__host__ __device__ void swap(Hittable **hlist,
                              int index_h1, int index_h2) {
  Hittable *temp = hlist[index_h1];
  hlist[index_h1] = hlist[index_h2];
  hlist[index_h2] = temp;
}

class BvhNode : public Hittable {
public:
  __device__ BvhNode();
  __device__ ~BvhNode() {
    delete left;
    delete right;
  }

  __device__ BvhNode(Hittables &hlist, float time0,
                     float time1)
      : BvhNode(hlist.list, hlist.list_size, time0, time1) {
  }

  __device__ BvhNode(Hittable **list, int list_size,
                     float time0, float time1);

  __device__ bool hit(const Ray &r, float tmin, float tmax,
                      HitRecord &rec) const override {
    if (!box.hit(r, t_min, t_max))
      return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right =
        right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
  }
  __device__ void odd_even_sort(Hittable **hlist,
                                int list_size) {
    bool sorted = false;
    while (!sorted) {
      sorted = true;
      for (int i = 1; i < list_size - 1; i += 2) {
        float d1 =
            distance_between_boxes(hlist[i - 1], hlist[i]);
        float d2 =
            distance_between_boxes(hlist[i + 1], hlist[i]);
        if (d2 < d1) {
          swap(hlist, i - 1, i + 1);
          sorted = false;
        }
      }
      for (int i = 2; i < list_size - 1; i += 2) {
        float d1 =
            distance_between_boxes(hlist[i - 1], hlist[i]);
        float d2 =
            distance_between_boxes(hlist[i + 1], hlist[i]);
        if (d2 < d1) {
          swap(hlist, i - 1, i + 1);
          sorted = false;
        }
      }
    }
  }

  __host__ __device__ float
  distance_between_boxes(Hittable *h1, Hittable *h2) {
    float h1_center;
    Aabb b1;
    if (!h1->bounding_box(time0, time1, b1)) {
      h1_center = h1->center;
    } else {
      h1_center = b1.center;
    }
    Aabb b2;
    if (!h2->bounding_box(time0, time1, b2)) {
      h2_center = h2->center;
    } else {
      h2_center = b2.center;
    }
    return distance(h1_center, h2_center);
  }
  __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {

    output_box = box;
    return true;
  }

public:
  Hittable *left;
  Hittable *right;
  Aabb box;
  float time0, time1;
};

struct Node {};

struct LeafNode : public Node {
  int objectID;
};
struct InternalNode : public Node {};

// from
// https://github.com/mbartling/cuda-bvh/blob/master/kernels/bvh.cu
__device__ void
determineRange(unsigned int *sortedMortonCodes,
               int numTriangles, int idx, int &left,
               int &right) {
  // determine the range of keys covered by each internal
  // node (as well as its children)
  // direction is found by looking at the neighboring keys
  // ki-1 , ki , ki+1
  // the index is either the beginning of the range or the
  // end of the range
  int direction = 0;
  int common_prefix_with_left = 0;
  int common_prefix_with_right = 0;

  common_prefix_with_right = __clz(
      sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
  if (idx == 0) {
    common_prefix_with_left = -1;
  } else {
    common_prefix_with_left =
        __clz(sortedMortonCodes[idx] ^
              sortedMortonCodes[idx - 1]);
  }

  direction = ((common_prefix_with_right -
                common_prefix_with_left) > 0)
                  ? 1
                  : -1;
  int min_prefix_range = 0;

  if (idx == 0) {
    min_prefix_range = -1;

  } else {
    min_prefix_range =
        __clz(sortedMortonCodes[idx] ^
              sortedMortonCodes[idx - direction]);
  }

  int lmax = 2;
  int next_key = idx + lmax * direction;

  while ((next_key >= 0) && (next_key < numTriangles) &&
         (__clz(sortedMortonCodes[idx] ^
                sortedMortonCodes[next_key]) >
          min_prefix_range)) {
    lmax *= 2;
    next_key = idx + lmax * direction;
  }
  // find the other end using binary search
  unsigned int l = 0;

  do {
    lmax = (lmax + 1) >> 1; // exponential decrease
    int new_val = idx + (l + lmax) * direction;

    if (new_val >= 0 && new_val < numTriangles) {
      unsigned int Code = sortedMortonCodes[new_val];
      int Prefix = __clz(sortedMortonCodes[idx] ^ Code);
      if (Prefix > min_prefix_range)
        l = l + lmax;
    }
  } while (lmax > 1);

  int j = idx + l * direction;

  int left = 0;
  int right = 0;

  if (idx < j) {
    left = idx;
    right = j;
  } else {
    left = j;
    right = idx;
  }
}

// all of the below from
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
__host__ __device__ int
findSplit(unsigned int *sortedMortonCodes, int first,
          int last) {
  // Identical Morton codes => split the range in the
  // middle.

  unsigned int firstCode = sortedMortonCodes[first];
  unsigned int lastCode = sortedMortonCodes[last];

  if (firstCode == lastCode)
    return (first + last) >> 1;

  // Calculate the number of highest bits that are the
  // same
  // for all objects, using the count-leading-zeros
  // intrinsic.

  int commonPrefix = __clz(firstCode ^ lastCode);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object
  // that
  // shares more than commonPrefix bits with the first
  // one.

  int split = first; // initial guess
  int step = last - first;

  do {
    step = (step + 1) >> 1;      // exponential decrease
    int newSplit = split + step; // proposed new position

    if (newSplit < last) {
      unsigned int splitCode = sortedMortonCodes[newSplit];
      int splitPrefix = __clz(firstCode ^ splitCode);
      if (splitPrefix > commonPrefix)
        split = newSplit; // accept proposal
    }
  } while (step > 1);

  return split;
}

__host__ __device__ Node *
generateHierarchy(unsigned int *sortedMortonCodes,
                  int *sortedObjectIDs, int numObjects) {
  LeafNode *leafNodes = new LeafNode[numObjects];
  InternalNode *internalNodes =
      new InternalNode[numObjects - 1];

  // Construct leaf nodes.
  // Note: This step can be avoided by storing
  // the tree in a slightly different way.

  for (int idx = 0; idx < numObjects; idx++) // in parallel
    leafNodes[idx].objectID = sortedObjectIDs[idx];

  // Construct internal nodes.

  for (int idx = 0; idx < numObjects - 1;
       idx++) // in parallel
  {
    // Find out which range of objects the node
    // corresponds
    // to.
    // (This is where the magic happens!)

    int2 range =
        determineRange(sortedMortonCodes, numObjects, idx);
    int first = range.x;
    int last = range.y;

    // Determine where to split the range.

    int split = findSplit(sortedMortonCodes, first, last);

    // Select childA.

    Node *childA;
    if (split == first)
      childA = &leafNodes[split];
    else
      childA = &internalNodes[split];

    // Select childB.

    Node *childB;
    if (split + 1 == last)
      childB = &leafNodes[split + 1];
    else
      childB = &internalNodes[split + 1];

    // Record parent-child relationships.

    internalNodes[idx].childA = childA;
    internalNodes[idx].childB = childB;
    childA->parent = &internalNodes[idx];
    childB->parent = &internalNodes[idx];
  }

  // Node 0 is the root.

  return &internalNodes[0];
}

class LBvhNode : public Hittable {};

#endif
