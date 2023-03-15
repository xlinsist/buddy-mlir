#!/bin/bash

BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLI := ../../llvm/build/bin/lli
LLC := ../../llvm/build/bin/llc
OPT := ../../llvm/build/bin/opt
OPT_FLAG := -O3
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

RISCV_GNU_TOOLCHAIN := ../../thirdparty/build-riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../thirdparty/build-riscv-gnu-toolchain/sysroot
QEMU := ../../thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64
LOCAL_CLANG := ../../thirdparty/build-local-clang/bin/clang
CROSS_LLI := ../../thirdparty/build-cross-clang/bin/lli
CROSS_MLIR_CPU_RUNNER := ../../thirdparty/build-cross-mlir/bin/mlir-cpu-runner
CROSS_MLIR_C_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_c_runner_utils.so
CROSS_MLIR_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_runner_utils.so
CROSS_MLIR_LIB := ../../thirdparty/build-cross-mlir/lib

SPLITING_SIZE := 128

conv2d-cb-lower-single-pass:
	@${BUDDY_OPT} ./conv2d.mlir \
		-conv-vectorization="strip-mining=${SPLITING_SIZE}" -o ./log.mlir

conv2d-cb-lower:
	@${BUDDY_OPT} ./conv2d.mlir \
		-conv-vectorization="strip-mining=${SPLITING_SIZE}" \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -llvm-request-c-wrappers \
		-convert-func-to-llvm -reconcile-unrealized-casts -o ./log.mlir

conv2d-cb-translate:
	@${BUDDY_OPT} ./conv2d.mlir \
		-conv-vectorization="strip-mining=${SPLITING_SIZE}" \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -llvm-request-c-wrappers \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o ./log.ll

conv2d-naive-asm:
	@${BUDDY_OPT} ./conv2d.mlir \
		-convert-linalg-to-loops \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -llvm-request-c-wrappers \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

conv2d-cb-masked-asm:
	@${BUDDY_OPT} ./conv2d-cb-masked.mlir \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -llvm-request-c-wrappers \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

conv2d-cb-stripmining-asm:
	@${BUDDY_OPT} ./conv2d-cb-stripmining.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-vector-exp \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

edge-detection-run-rv: conv2d-naive-rvv.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ -c -I ../../frontend/Interfaces/ `pkg-config opencv4 --cflags` edge-detection.cpp `pkg-config opencv4 --libs`
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ `pkg-config opencv4 --cflags` conv2d-naive-rvv.o edge-detection.o `pkg-config opencv4 --libs`\
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -no-pie -mabi=lp64d -fPIC -o edge-detection
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 edge-detection ../images/YuTu.png

edge-detection-compare-run-rv: conv2d-naive-rvv.o conv2d-cb-masked-rvv.o conv2d-cb-stripmining-rvv.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ -c -I ../../frontend/Interfaces/ `pkg-config opencv4 --cflags` edge-detection-compare.cpp `pkg-config opencv4 --libs`
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ `pkg-config opencv4 --cflags` conv2d-naive-rvv.o conv2d-cb-masked-rvv.o conv2d-cb-stripmining-rvv.o edge-detection-compare.o `pkg-config opencv4 --libs`\
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -no-pie -mabi=lp64d -fPIC -o edge-detection-compare
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 edge-detection-compare ../images/YuTu.png

conv2d-naive-rvv.o:
	@${BUDDY_OPT} ./conv2d.mlir \
		-convert-linalg-to-loops \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -llvm-request-c-wrappers \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -O3 -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -filetype=obj -riscv-v-vector-bits-min=128 \
		-o ./conv2d-naive-rvv.o

conv2d-cb-masked-rvv.o:
	@${BUDDY_OPT} ./conv2d-cb-masked.mlir \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -llvm-request-c-wrappers \
		-convert-func-to-llvm -reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -O3 -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -filetype=obj -riscv-v-vector-bits-min=128 \
		-o ./conv2d-cb-masked-rvv.o

conv2d-cb-stripmining-rvv.o:
	@${BUDDY_OPT} ./conv2d-cb-stripmining.mlir \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-math-to-llvm \
		-lower-vector-exp \
		-lower-rvv \
		-convert-vector-to-llvm \
		-convert-memref-to-llvm \
		-llvm-request-c-wrappers \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${LLC} -O3 -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -filetype=obj -riscv-v-vector-bits-min=128 \
		-o ./conv2d-cb-stripmining-rvv.o

.PHONY: clean
clean:
	rm log.mlir conv2d-rvv.o edge-detection.o edge-detection buddy-result.png opencv-result.png \
	edge-detection-compare.o edge-detection-compare conv2d-naive-rvv.o conv2d-cb-masked-rvv.o conv2d-cb-stripmining-rvv.o
