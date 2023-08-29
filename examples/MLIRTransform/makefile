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

transform-batch-matmul-vectorize-lower:
	@${MLIR_OPT} ./transform-batch-matmul-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		-o log.mlir

transform-batch-matmul-vectorize-x86-asm:
	@${MLIR_OPT} ./transform-batch-matmul-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
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
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f\
		--filetype=asm -o log.s

transform-batch-matmul-vectorize-rv-asm:
	@${MLIR_OPT} ./transform-batch-matmul-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
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
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

transform-batch-matmul-vectorize-x86-o:
	@${MLIR_OPT} ./transform-batch-matmul-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu \
		-mattr=avx512f --filetype=obj \
	    -o ./transform-batch-matmul-vectorize.o

transform-batch-matmul-vectorize-rv-o:
	@${MLIR_OPT} ./transform-batch-matmul-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -filetype=obj -riscv-v-vector-bits-min=128 \
		-o ./transform-batch-matmul-vectorize.o

transform-linalg-generic-vectorize-lower:
	@${MLIR_OPT} ./transform-linalg-generic-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		-o log.mlir

transform-linalg-generic-vectorize-rv-asm:
	@${MLIR_OPT} ./transform-linalg-generic-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
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
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

transform-conv2d-im2col-vectorize-lower:
	@${MLIR_OPT} ./transform-conv2d-im2col-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		-o log.mlir

# TODO: Debug why the lowering process will get stuck in `--convert-linalg-to-llvm`.
transform-conv2d-im2col-vectorize-rv-asm:
	@${MLIR_OPT} ./transform-conv2d-im2col-vectorize.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-linalg-to-loops \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
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
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s
