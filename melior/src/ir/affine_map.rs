use crate::{
    affine_expr::AffineExpr,
    context::{Context, ContextRef},
    utility::print_callback,
};
use mlir_sys::{
    mlirAffineMapConstantGet, mlirAffineMapDump, mlirAffineMapEmptyGet, mlirAffineMapEqual,
    mlirAffineMapGet, mlirAffineMapGetContext, mlirAffineMapGetMajorSubMap,
    mlirAffineMapGetMinorSubMap, mlirAffineMapGetNumDims, mlirAffineMapGetNumInputs,
    mlirAffineMapGetNumResults, mlirAffineMapGetNumSymbols, mlirAffineMapGetResult,
    mlirAffineMapGetSingleConstantResult, mlirAffineMapGetSubMap, mlirAffineMapIsEmpty,
    mlirAffineMapIsIdentity, mlirAffineMapIsMinorIdentity, mlirAffineMapIsPermutation,
    mlirAffineMapIsProjectedPermutation, mlirAffineMapIsSingleConstant,
    mlirAffineMapMinorIdentityGet, mlirAffineMapMultiDimIdentityGet, mlirAffineMapPermutationGet,
    mlirAffineMapPrint, mlirAffineMapReplace, mlirAffineMapZeroResultGet, MlirAffineMap,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// An affine map.
#[derive(Clone, Copy)]
pub struct AffineMap<'c> {
    raw: MlirAffineMap,
    _context: PhantomData<&'c Context>,
}

impl<'c> AffineMap<'c> {
    pub fn to_raw(&self) -> MlirAffineMap {
        self.raw
    }

    pub fn from(
        context: &Context,
        dim_count: isize,
        symbol_count: isize,
        affine_exprs: Vec<AffineExpr>,
    ) -> Self {
        unsafe {
            let mut expr_ptr: Vec<mlir_sys::MlirAffineExpr> =
                affine_exprs.iter().map(|e| e.to_raw()).collect();
            let map = mlirAffineMapGet(
                context.to_raw(),
                dim_count,
                symbol_count,
                affine_exprs.len() as isize,
                expr_ptr.as_mut_ptr(),
            );

            AffineMap::from_raw(map)
        }
    }

    /// Creates a single constant result affine map in the context. The affine map
    /// is owned by the context.
    pub fn from_constant(context: &Context, val: i64) -> Self {
        unsafe { AffineMap::from_raw(mlirAffineMapConstantGet(context.to_raw(), val)) }
    }

    /// Creates an affine map with 'numDims' identity in the context. The affine map
    /// is owned by the context.
    pub fn from_multi_dim_identity(context: &Context, num_dims: isize) -> Self {
        unsafe { AffineMap::from_raw(mlirAffineMapMultiDimIdentityGet(context.to_raw(), num_dims)) }
    }

    /// Creates an identity affine map on the most minor dimensions in the context.
    /// The affine map is owned by the context. The function asserts that the number
    /// of dimensions is greater or equal to the number of results.
    pub fn from_minor_identity(context: &Context, dims: isize, results: isize) -> Self {
        unsafe {
            AffineMap::from_raw(mlirAffineMapMinorIdentityGet(
                context.to_raw(),
                dims,
                results,
            ))
        }
    }

    /// Creates an affine map with a permutation expression and its size in the
    /// context. The permutation expression is a non-empty vector of integers.
    /// The elements of the permutation vector must be continuous from 0 and cannot
    /// be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is
    /// an invalid permutation.) The affine map is owned by the context.
    pub fn from_permutation(context: &Context, size: isize, permutation: Vec<u32>) -> Self {
        unsafe {
            let permutation = permutation.clone().as_mut_ptr();
            AffineMap::from_raw(mlirAffineMapPermutationGet(
                context.to_raw(),
                size,
                permutation,
            ))
        }
    }

    /// Creates a zero result affine map with no dimensions or symbols in the
    /// context. The affine map is owned by the context.
    pub fn empty(context: &Context) -> Self {
        unsafe { AffineMap::from_raw(mlirAffineMapEmptyGet(context.to_raw())) }
    }

    /// Creates a zero result affine map of the given dimensions and symbols in the
    /// context. The affine map is owned by the context.
    pub fn zero_result(context: &Context, dim_count: isize, symbol_count: isize) -> Self {
        unsafe {
            AffineMap::from_raw(mlirAffineMapZeroResultGet(
                context.to_raw(),
                dim_count,
                symbol_count,
            ))
        }
    }

    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAffineMapGetContext(self.raw)) }
    }

    /// Dumps an affine map.
    pub fn dump(&self) {
        unsafe { mlirAffineMapDump(self.raw) }
    }

    /// Creates an affine map from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirAffineMap) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    pub fn is_identity(&self) -> bool {
        unsafe { mlirAffineMapIsIdentity(self.raw) }
    }

    pub fn is_minor_identity(&self) -> bool {
        unsafe { mlirAffineMapIsMinorIdentity(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        unsafe { mlirAffineMapIsEmpty(self.raw) }
    }

    pub fn is_single_constant(&self) -> bool {
        unsafe { mlirAffineMapIsSingleConstant(self.raw) }
    }

    pub fn to_single_constant_result(&self) -> i64 {
        unsafe { mlirAffineMapGetSingleConstantResult(self.raw) }
    }

    pub fn num_dimensions(&self) -> isize {
        unsafe { mlirAffineMapGetNumDims(self.raw) }
    }

    pub fn num_results(&self) -> isize {
        unsafe { mlirAffineMapGetNumResults(self.raw) }
    }

    pub fn num_symbols(&self) -> isize {
        unsafe { mlirAffineMapGetNumSymbols(self.raw) }
    }

    /// Returns the number of inputs (dimensions + symbols) of the given affine
    /// map.
    pub fn num_inputs(&self) -> isize {
        unsafe { mlirAffineMapGetNumInputs(self.raw) }
    }

    pub fn get_result(&self, pos: isize) -> AffineExpr {
        unsafe { AffineExpr::from_raw(mlirAffineMapGetResult(self.raw, pos)) }
    }

    /// Checks whether the given affine map represents a subset of a symbol-less
    /// permutation map.
    pub fn is_projected_permutation(&self) -> bool {
        unsafe { mlirAffineMapIsProjectedPermutation(self.raw) }
    }

    /// Checks whether the given affine map represents a symbol-less permutation
    /// map.
    pub fn is_permutation(&self) -> bool {
        unsafe { mlirAffineMapIsPermutation(self.raw) }
    }

    /// Returns the affine map consisting of the `resultPos` subset.
    pub fn sub_map(&self, size: isize, result_pos: Vec<isize>) -> Self {
        let result_pos_ptr = result_pos.clone().as_mut_ptr();
        unsafe { Self::from_raw(mlirAffineMapGetSubMap(self.raw, size, result_pos_ptr)) }
    }

    /// Returns the affine map consisting of the most major `numResults` results.
    /// Returns the null AffineMap if the `numResults` is equal to zero.
    /// Returns the `affineMap` if `numResults` is greater or equals to number of
    /// results of the given affine map.
    pub fn major_sub_map(&self, num_results: isize) -> Self {
        unsafe { Self::from_raw(mlirAffineMapGetMajorSubMap(self.raw, num_results)) }
    }

    /// Returns the affine map consisting of the most minor `numResults` results.
    /// Returns the null AffineMap if the `numResults` is equal to zero.
    /// Returns the `affineMap` if `numResults` is greater or equals to number of
    /// results of the given affine map.
    pub fn minor_sub_map(&self, num_results: isize) -> Self {
        unsafe { Self::from_raw(mlirAffineMapGetMinorSubMap(self.raw, num_results)) }
    }

    /// Apply AffineExpr::replace(`map`) to each of the results and return a new
    /// new AffineMap with the new results and the specified number of dims and
    /// symbols.
    pub fn replace(
        &self,
        expression: AffineExpr,
        replacement: AffineExpr,
        num_result_dims: isize,
        num_result_symbols: isize,
    ) -> Self {
        unsafe {
            Self::from_raw(mlirAffineMapReplace(
                self.raw,
                expression.to_raw(),
                replacement.to_raw(),
                num_result_dims,
                num_result_symbols,
            ))
        }
    }

    // TODO: mlirAffineMapCompressUnusedSymbols
}

impl<'c> PartialEq for AffineMap<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAffineMapEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for AffineMap<'c> {}

impl<'c> Display for AffineMap<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirAffineMapPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'c> Debug for AffineMap<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
