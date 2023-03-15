//====- edge-detection-compare.cpp - Example of buddy-opt tool --------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file compares performances with conv CB algorithm between masked version
// and strip-mining version.
//
//===----------------------------------------------------------------------===//

#include "../kernels.h"
#include <buddy/DIP/ImageContainer.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <time.h>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.

extern "C" {
void _mlir_ciface_conv_2d(Img<float, 2> *input, MemRef<float, 2> *kernel,
                          MemRef<float, 2> *output);
}

extern "C" {
void _mlir_ciface_conv2d_cb_masked(Img<float, 2> *input,
                                   MemRef<float, 2> *kernel,
                                   MemRef<float, 2> *output);
}

extern "C" {
void _mlir_ciface_conv2d_cb_stripmining(Img<float, 2> *input,
                                        MemRef<float, 2> *kernel,
                                        MemRef<float, 2> *output);
}

Mat buddyInputMat;
int kernelRows;
int kernelCols;

void PerformConv2DCBMasked(Img<float, 2> buddyInputMemRef,
                           MemRef<float, 2> kernelMemRef, int method) {
  int turn = 10;
  double totalBuddyConv2DTime;

  int outputRows = buddyInputMat.rows - kernelRows + 1;
  int outputCols = buddyInputMat.cols - kernelCols + 1;
  intptr_t sizesOutput[2] = {outputRows, outputCols};
  while (turn--) {
    MemRef<float, 2> outputMemRef(sizesOutput);

    /// Evaluate the Buddy Conv2D.
    clock_t buddyConvStart;
    clock_t buddyConvEnd;

    if (method == 0) {
      buddyConvStart = clock();
      // Perform the Conv2D function.
      _mlir_ciface_conv_2d(&buddyInputMemRef, &kernelMemRef, &outputMemRef);
      buddyConvEnd = clock();
    } else if (method == 1) {
      buddyConvStart = clock();
      // Perform the Conv2D function.
      _mlir_ciface_conv2d_cb_masked(&buddyInputMemRef, &kernelMemRef,
                                    &outputMemRef);
      buddyConvEnd = clock();
    } else if (method == 2) {
      buddyConvStart = clock();
      // Perform the Conv2D function.
      _mlir_ciface_conv2d_cb_stripmining(&buddyInputMemRef, &kernelMemRef,
                                         &outputMemRef);
      buddyConvEnd = clock();
    } else
      assert(false);

    double buddyConv2DTime =
        (double)(buddyConvEnd - buddyConvStart) / CLOCKS_PER_SEC;
    totalBuddyConv2DTime += buddyConv2DTime;
  }
  double averageBuddyConv2DTime = totalBuddyConv2DTime / 10.0;
  std::vector<std::string> methodName = {"naive", "CB masked",
                                         "CB stripmining"};
  cout << "[" << methodName[method]
       << "] Perform Conv2D time: " << averageBuddyConv2DTime << " s" << endl;
}
int main(int argc, char *argv[]) {
  cout << "Start processing..." << endl;
  cout << "-----------------------------------------" << endl;
  //-------------------------------------------------------------------------//
  // Buddy Conv2D
  //-------------------------------------------------------------------------//

  /// Evaluate Buddy reading input process.
  clock_t buddyReadStart;
  buddyReadStart = clock();
  // Read as grayscale image.
  buddyInputMat = imread(argv[1], IMREAD_GRAYSCALE);
  Img<float, 2> buddyInputMemRef(buddyInputMat);
  clock_t buddyReadEnd;
  buddyReadEnd = clock();
  double buddyReadTime =
      (double)(buddyReadEnd - buddyReadStart) / CLOCKS_PER_SEC;
  cout << "[Buddy] Read input time: " << buddyReadTime << " s" << endl;

  /// Evaluate Buddy defining kernel process.
  clock_t buddyKernelStart;
  buddyKernelStart = clock();
  // Get the data, row, and column information of the kernel.
  float *kernelAlign = laplacianKernelAlign;
  kernelRows = laplacianKernelRows;
  kernelCols = laplacianKernelCols;
  intptr_t sizesKernel[2] = {kernelRows, kernelCols};
  // Define the kernel MemRef object.
  MemRef<float, 2> kernelMemRef(kernelAlign, sizesKernel);
  clock_t buddyKernelEnd;
  buddyKernelEnd = clock();
  double buddyKernelTime =
      (double)(buddyKernelEnd - buddyKernelStart) / CLOCKS_PER_SEC;
  cout << "[Buddy] Define kernel time: " << buddyKernelTime << " s" << endl;

  PerformConv2DCBMasked(buddyInputMemRef, kernelMemRef, 0);
  PerformConv2DCBMasked(buddyInputMemRef, kernelMemRef, 1);
  PerformConv2DCBMasked(buddyInputMemRef, kernelMemRef, 2);

  cout << "-----------------------------------------" << endl;
  return 0;
}
