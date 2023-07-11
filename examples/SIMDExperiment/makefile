#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

conv2d-nchw-fchw-im2col-lower:
	@${MLIR_OPT} ./conv2d-nchw-fchw-im2col.mlir \
	--convert-vector-to-scf \
	--convert-scf-to-cf \
	--expand-strided-metadata \
	--lower-affine \
	--convert-vector-to-llvm \
	--memref-expand \
	--arith-expand \
	--convert-arith-to-llvm \
	--finalize-memref-to-llvm \
	--convert-math-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts \
	-o ./log.mlir

conv2d-nchw-fchw-im2col-translate:
	@${MLIR_OPT} ./conv2d-nchw-fchw-im2col.mlir \
	--convert-vector-to-scf \
	--convert-scf-to-cf \
	--expand-strided-metadata \
	--lower-affine \
	--convert-vector-to-llvm \
	--memref-expand \
	--arith-expand \
	--convert-arith-to-llvm \
	--finalize-memref-to-llvm \
	--convert-math-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

conv2d-nchw-fchw-im2col-run:
	@${BUDDY_OPT} ./conv2d-nchw-fchw-im2col.mlir \
	--convert-vector-to-scf \
	--convert-scf-to-cf \
	--expand-strided-metadata \
	--lower-affine \
	--convert-vector-to-llvm \
	--memref-expand \
	--arith-expand \
	--convert-arith-to-llvm \
	--finalize-memref-to-llvm \
	--convert-math-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

conv2d-nchw-fchw-vectorize-16-lower:
	@${MLIR_OPT} ./conv2d-nchw-fchw-vectorize-16.mlir \
	--lower-affine \
	--convert-vector-to-llvm \
	--finalize-memref-to-llvm \
	--convert-scf-to-cf \
	--convert-linalg-to-llvm \
	--llvm-request-c-wrappers \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts \
	-o ./log.mlir

conv2d-nchw-fchw-vectorize-16-translate:
	@${MLIR_OPT} ./conv2d-nchw-fchw-vectorize-16.mlir \
	--lower-affine \
	--convert-vector-to-llvm \
	--finalize-memref-to-llvm \
	--convert-scf-to-cf \
	--convert-linalg-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

conv2d-nchw-fchw-vectorize-16-run:
	@${MLIR_OPT} ./conv2d-nchw-fchw-vectorize-16.mlir \
	--lower-affine \
	--convert-vector-to-llvm \
	--finalize-memref-to-llvm \
	--convert-scf-to-cf \
	--convert-linalg-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

winagrad-lower:
	@${MLIR_OPT} ./winagrad-F23.mlir \
	--convert-linalg-to-loops \
	--expand-strided-metadata \
	--lower-affine \
	--convert-vector-to-scf \
	--convert-scf-to-cf \
	--convert-vector-to-llvm \
	--memref-expand \
	--arith-expand \
	--convert-arith-to-llvm \
	--convert-index-to-llvm \
	--finalize-memref-to-llvm \
	--convert-math-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts \
	-o ./log.mlir

winagrad-translate:
	@${MLIR_OPT} ./winagrad-F23.mlir \
	--convert-linalg-to-loops \
	--expand-strided-metadata \
	--lower-affine \
	--convert-vector-to-scf \
	--convert-scf-to-cf \
	--convert-vector-to-llvm \
	--memref-expand \
	--arith-expand \
	--convert-arith-to-llvm \
	--convert-index-to-llvm \
	--finalize-memref-to-llvm \
	--convert-math-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

winagrad-run:
	@${MLIR_OPT} ./winagrad-F63.mlir \
	--convert-linalg-to-loops \
	--expand-strided-metadata \
	--lower-affine \
	--convert-vector-to-scf \
	--convert-scf-to-cf \
	--convert-vector-to-llvm \
	--memref-expand \
	--arith-expand \
	--convert-arith-to-llvm \
	--convert-index-to-llvm \
	--finalize-memref-to-llvm \
	--convert-math-to-llvm \
	--convert-func-to-llvm \
	--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

# TODO: fix bugs concerning `builtin.unrealized_conversion_cast` for matmul-vectorize-16-lower.
matmul-vectorize-16-lower:
	@${BUDDY_OPT} ../../../buddy-benchmark/benchmarks/OpOptimization/MatMul/gemm.mlir \
	--matmul-optimize="vec-size=16" \
	--lower-affine \
	-convert-vector-to-llvm \
	-finalize-memref-to-llvm \
	-convert-scf-to-cf \
	-convert-linalg-to-llvm \
	-convert-func-to-llvm \
	-reconcile-unrealized-casts \
	-o ./log.mlir

test-lower:
	@${MLIR_OPT} ./test.mlir \
	--test-transform-dialect-interpreter \
	--lower-affine \
	-convert-vector-to-llvm \
	-finalize-memref-to-llvm \
	-convert-scf-to-cf \
	-convert-linalg-to-llvm \
	-convert-func-to-llvm \
	-reconcile-unrealized-casts \
	-o ./log.mlir

test-help:
	@${MLIR_OPT} -h
