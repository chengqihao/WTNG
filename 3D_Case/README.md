# WTNG: Efficient Multi-Vector Search via Weight-Template Navigation Graph

All codes are implemented in C++17

## Requisites

* GCC 4.9+ with OpenMP
* CMake 2.8+
* Boost 1.55+

## Code Running 

### Datasets

Our experiment involves four real-world datasets which can be downloaded from the link in the paper. 


### Compile on Linux

```shell
$ mkdir build && cd build/
$ cmake ..
$ make -j
```

### Index construction 

An example of index construction

```shell
cd ./build/test/
./main algorithm_name dataset_name \alpha_1 \alpha_2  max_distance_1 max_distance_2 max_distance_3 build
```

### Search 

An example of search

```shell
cd ./build/test/
./main algorithm_name dataset_name \alpha_1 \alpha_2 max_distance_1 max_distance_2  max_distance_3 search
```
