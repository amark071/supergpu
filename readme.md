```shell
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libmetis-dev \
    libsuitesparse-dev \
    python3 \
    python3-pip
```

```cpp
python3 -m pip install matplotlib networkx
```

```cpp
cmake -S . -B build-sm120 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build-sm120 -j$(nproc)
```

```cpp
./build/supernodal_gpu
```

