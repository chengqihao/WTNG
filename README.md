# WTNG

WTNG is an experimental C++ implementation for Paper: ``WTNG: Efficient Multi-Vector Search via
Weight-Template Navigation Graph''


## Implemented Algorithms

The executable accepts the following algorithm names:

- `wtng`: builds and searches the WTNG graph.
- `baseline1`: builds one HNSW graph at a fixed balanced weight.
- `baseline2`: builds three HNSW graphs, one for each modality-dominant corner.
- `vbase`: builds a multi-index baseline for the three modalities.


## Dependencies

Required:

- C++17 compiler
- CMake 3.10+
- OpenMP
- Boost


## Datasets

Our experiment involves four real-world datasets, all of them can be downloaded from the link in the paper. Note that, all base data and query data are converted to fvecs format, and ground-truth data is converted to ivecs format. 


## Build

From the project root:

```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

## Command Line Usage

```bash
./build/test/main \
  <algorithm> \
  <dataset> \
  <alpha1> \
  <alpha2> \
  <max_spatial_distance> \
  <max_emb_distance> \
  <max_video_distance> \
  <build|search>
```

Example: build and search WTNG on openimage.

```bash
./build/test/main wtng openimage 0.7 0.1 1 1 1 build
./build/test/main wtng openimage 0.3 0.3 1 1 1 search
```

