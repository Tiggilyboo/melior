use std::marker::PhantomData;

use mlir_sys::{
    self, mlirAffineAddExprGet, mlirAffineBinaryOpExprGetLHS, mlirAffineBinaryOpExprGetRHS,
    mlirAffineCeilDivExprGet, mlirAffineConstantExprGet, mlirAffineConstantExprGetValue,
    mlirAffineDimExprGet, mlirAffineDimExprGetPosition, mlirAffineExprCompose, mlirAffineExprDump,
    mlirAffineExprEqual, mlirAffineExprGetContext, mlirAffineExprGetLargestKnownDivisor,
    mlirAffineExprIsAAdd, mlirAffineExprIsABinary, mlirAffineExprIsACeilDiv,
    mlirAffineExprIsAConstant, mlirAffineExprIsADim, mlirAffineExprIsAFloorDiv,
    mlirAffineExprIsAMod, mlirAffineExprIsAMul, mlirAffineExprIsASymbol,
    mlirAffineExprIsFunctionOfDim, mlirAffineExprIsMultipleOf, mlirAffineExprIsPureAffine,
    mlirAffineExprIsSymbolicOrConstant, mlirAffineFloorDivExprGet, mlirAffineModExprGet,
    mlirAffineMulExprGet, MlirAffineExpr,
};

use crate::{ir::AffineMap, Context, ContextRef};

#[derive(Debug, Copy, Clone)]
pub struct AffineExpr<'c> {
    raw: MlirAffineExpr,
    _context: PhantomData<&'c Context>,
}

impl<'c> AffineExpr<'c> {
    pub fn to_raw(&self) -> MlirAffineExpr {
        self.raw
    }

    pub unsafe fn from_raw(raw: MlirAffineExpr) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
    pub fn from_map(&self, affine_map: AffineMap) -> Self {
        unsafe { Self::from_raw(mlirAffineExprCompose(self.raw, affine_map.to_raw())) }
    }
    pub fn context(&self) -> ContextRef {
        unsafe { ContextRef::from_raw(mlirAffineExprGetContext(self.raw)) }
    }

    /// Prints the affine expression to the standard error stream.
    pub fn dump(&self) {
        unsafe { mlirAffineExprDump(self.raw) }
    }

    /// Checks whether the given affine expression is made out of only symbols and
    /// constants.
    pub fn is_symbolic_or_constant(&self) -> bool {
        unsafe { mlirAffineExprIsSymbolicOrConstant(self.raw) }
    }

    /// Checks whether the given affine expression is a pure affine expression, i.e.
    /// mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
    pub fn is_pure_affine(&self) -> bool {
        unsafe { mlirAffineExprIsPureAffine(self.raw) }
    }

    /// Returns the greatest known integral divisor of this affine expression. The
    /// result is always positive.
    pub fn largest_known_divisor(&self) -> i64 {
        unsafe { mlirAffineExprGetLargestKnownDivisor(self.raw) }
    }

    /// Checks whether the given affine expression is a multiple of 'factor'.
    pub fn is_multiple_of(&self, factor: i64) -> bool {
        unsafe { mlirAffineExprIsMultipleOf(self.raw, factor) }
    }

    pub fn is_function_of_dimension(&self, position: isize) -> bool {
        unsafe { mlirAffineExprIsFunctionOfDim(self.raw, position) }
    }

    pub fn is_dimension(&self) -> bool {
        unsafe { mlirAffineExprIsADim(self.raw) }
    }

    /// Creates an affine dimension expression with 'position' in the context.
    pub fn new_dimension(context: &Context, position: isize) -> Self {
        unsafe { Self::from_raw(mlirAffineDimExprGet(context.to_raw(), position)) }
    }

    /// Returns the position of the given affine dimension expression.
    pub fn dimension_position(&self) -> isize {
        unsafe { mlirAffineDimExprGetPosition(self.raw) }
    }

    pub fn is_symbol(&self) -> bool {
        unsafe { mlirAffineExprIsASymbol(self.raw) }
    }

    /// Checks whether the given affine expression is a constant expression.
    pub fn is_constant(&self) -> bool {
        unsafe { mlirAffineExprIsAConstant(self.raw) }
    }

    /// Creates an affine constant expression with 'constant' in the context.
    pub fn new_constant(context: &Context, constant: i64) -> Self {
        unsafe { Self::from_raw(mlirAffineConstantExprGet(context.to_raw(), constant)) }
    }

    /// Returns the value of the given affine constant expression.
    pub fn constant_value(&self) -> i64 {
        unsafe { mlirAffineConstantExprGetValue(self.raw) }
    }

    /// Checks whether the given affine expression is an add expression.
    pub fn is_add(&self) -> bool {
        unsafe { mlirAffineExprIsAAdd(self.raw) }
    }

    /// Creates an affine add expression with 'lhs' and 'rhs'.
    pub fn add(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineAddExprGet(lhs.to_raw(), rhs.to_raw())) }
    }

    /// Checks whether the given affine expression is an mul expression.
    pub fn is_mul(&self) -> bool {
        unsafe { mlirAffineExprIsAMul(self.raw) }
    }

    /// Creates an affine mul expression with 'lhs' and 'rhs'.
    pub fn multiply(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineMulExprGet(lhs.to_raw(), rhs.to_raw())) }
    }

    /// Checks whether the given affine expression is a mod expression.
    pub fn is_mod(&self) -> bool {
        unsafe { mlirAffineExprIsAMod(self.raw) }
    }

    /// Creates an affine mul expression with 'lhs' and 'rhs'.
    pub fn modulus(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineModExprGet(lhs.to_raw(), rhs.to_raw())) }
    }

    /// Checks whether the given affine expression is a floor division expression.
    pub fn is_floor_div(&self) -> bool {
        unsafe { mlirAffineExprIsAFloorDiv(self.raw) }
    }

    /// Creates an affine floor division expression with 'lhs' and 'rhs'.
    pub fn floor_div(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineFloorDivExprGet(lhs.to_raw(), rhs.to_raw())) }
    }

    /// Checks whether the given affine expression is a ceiling division expression.
    pub fn is_ceil_div(&self) -> bool {
        unsafe { mlirAffineExprIsACeilDiv(self.raw) }
    }

    /// Creates an affine ceiling division expression with 'lhs' and 'rhs'.
    pub fn ceil_div(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineCeilDivExprGet(lhs.to_raw(), rhs.to_raw())) }
    }

    /// Checks whether the given affine expression is a binary expression.
    pub fn is_binary(&self) -> bool {
        unsafe { mlirAffineExprIsABinary(self.raw) }
    }

    /// Returns the left hand side affine expression of the given affine binary
    /// operation expression.
    pub fn binary_lhs(&self) -> Self {
        unsafe { Self::from_raw(mlirAffineBinaryOpExprGetLHS(self.raw)) }
    }

    /// Returns the left hand side affine expression of the given affine binary
    /// operation expression.
    pub fn binary_rhs(&self) -> Self {
        unsafe { Self::from_raw(mlirAffineBinaryOpExprGetRHS(self.raw)) }
    }
}

impl<'c> Eq for AffineExpr<'c> {}

impl<'c> PartialEq for AffineExpr<'c> {
    fn eq(&self, other: &AffineExpr) -> bool {
        unsafe { mlirAffineExprEqual(self.raw, other.raw) }
    }
}
