//! Sparse tensor passes.

melior_macro::passes!(
    "SparseTensor",
    [
        mlirCreateSparseTensorLowerForeachToSCF,
        mlirCreateSparseTensorLowerSparseOpsToForeach,
        mlirCreateSparseTensorPreSparsificationRewrite,
        mlirCreateSparseTensorSparseBufferRewrite,
        mlirCreateSparseTensorSparseTensorCodegen,
        mlirCreateSparseTensorSparseTensorConversionPass,
        mlirCreateSparseTensorSparseReinterpretMap,
        mlirCreateSparseTensorSparseVectorization,
        mlirCreateSparseTensorSparsificationPass,
        mlirCreateSparseTensorStorageSpecifierToLLVM,
        mlirCreateSparseTensorSparsificationAndBufferization,
        mlirCreateSparseTensorStageSparseOperations
    ]
);
