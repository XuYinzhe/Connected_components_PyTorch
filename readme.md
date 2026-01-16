# Connected Components Labeling for PyTorch

Update notes 2024.12
- Fix bugs and test in PyTorch version 1.13.1+cu117 and 2.1.2+cu118
- Installation: `python setup.py install`  

Update notes 2026.01
- Simplify code structure for c++ compilation
  
`CMakeLists.txt` examples
```cmake
set(CC_TORCH_SRC_DIR ${PROJECT_SOURCE_DIR}/third_party/Connected_components_PyTorch/cpp)

add_library(cc_torch SHARED
    ${CC_TORCH_SRC_DIR}/buf.h
    ${CC_TORCH_SRC_DIR}/buf.cu)

set_target_properties(cc_torch PROPERTIES CUDA_ARCHITECTURES "75;86")

target_include_directories(cc_torch PUBLIC
    ${CC_TORCH_SRC_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cc_torch PUBLIC
    ${TORCH_LIBRARIES}
)
```
Usage in c++
```cpp
#include "third_party/Connected_components_PyTorch/cpp/buf.h"

auto labeled_mask = cc_torch::connected_components_labeling_2d(mask.to(torch::kUInt8));
```

## References
- [YACCLAB : Yet Another Connected Components Labeling Benchmark](https://github.com/prittt/YACCLAB)

- <p align="justify"> Allegretti, Stefano; Bolelli, Federico; Grana, Costantino "Optimized Block-Based Algorithms to Label Connected Components on GPUs." IEEE Transactions on Parallel and Distributed Systems, 2019. <a title="BibTex" href="http://imagelab.ing.unimore.it/files2/yacclab/YACCLAB_TPDS2019_BibTex.html">BibTex</a>. <a title="Download" href="https://iris.unimore.it/retrieve/handle/11380/1179616/225393/2018_TPDS_Optimized_Block_Based_Algorithms_to_Label_Connected_Components_on_GPUs.pdf"><img src="https://raw.githubusercontent.com/prittt/YACCLAB/master/doc/pdf_logo.png" alt="Download." /></a></p>

<hr>

Follwing Block-based Union Find Algorithm from YACCLAB.

 - Running on GPU.
 - PyTorch Interface
 - Fix some bit alignment problem
 - little refactoring


## Example

Tested on [scikit-image example](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html) ,follows [example.ipynb](example.ipynb)

![img.png](https://user-images.githubusercontent.com/18730255/95699234-e60daa80-0c7e-11eb-9a1d-6059586c1899.png)

## Install

```
> python3 setup.py install

> python3 setup.py test
```
