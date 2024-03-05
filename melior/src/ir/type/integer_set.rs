use std::marker::PhantomData;

use mlir_sys::{
    mlirIntegerSetEmptyGet, mlirIntegerSetEqual, mlirIntegerSetGet, mlirIntegerSetGetConstraint,
    mlirIntegerSetGetNumConstraints, mlirIntegerSetGetNumDims, mlirIntegerSetGetNumEqualities,
    mlirIntegerSetGetNumInequalities, mlirIntegerSetGetNumInputs, mlirIntegerSetGetNumSymbols,
    mlirIntegerSetIsCanonicalEmpty, mlirIntegerSetIsConstraintEq, MlirAffineExpr, MlirIntegerSet,
};

use crate::{affine_expr::AffineExpr, Context};

#[derive(Clone, Copy, Debug)]
pub struct IntegerSet<'c> {
    raw: MlirIntegerSet,
    _context: PhantomData<&'c Context>,
}

#[allow(dead_code)]
impl<'c> IntegerSet<'c> {
    // Gets or creates a new integer set in the given context. The set is defined
    /// by a list of affine constraints, with the given number of input dimensions
    /// and symbols, which are treated as either equalities (eqFlags is 1) or
    /// inequalities (eqFlags is 0). Both `constraints` and `eqFlags` are expected
    /// to point to at least `numConstraint` consecutive values.
    pub fn new(
        context: &'c Context,
        num_dims: isize,
        num_symbols: isize,
        constraints: Vec<MlirAffineExpr>,
        eq_flags: bool,
    ) -> Self {
        unsafe {
            let int_set = mlirIntegerSetGet(
                context.to_raw(),
                num_dims,
                num_symbols,
                constraints.len() as isize,
                constraints.as_ptr(),
                &eq_flags,
            );

            Self {
                raw: int_set,
                _context: Default::default(),
            }
        }
    }

    pub fn to_raw(&self) -> MlirIntegerSet {
        self.raw
    }

    /// Gets or creates a new canonically empty integer set with the give number of
    /// dimensions and symbols in the given context.
    pub fn empty(context: &'c Context, num_dims: isize, num_symbols: isize) -> Self {
        unsafe {
            let int_set = mlirIntegerSetEmptyGet(context.to_raw(), num_dims, num_symbols);

            Self {
                raw: int_set,
                _context: Default::default(),
            }
        }
    }

    /// Returns the number of constraints (equalities + inequalities) in the given
    /// set.
    pub fn num_constraints(&self) -> isize {
        unsafe { mlirIntegerSetGetNumConstraints(self.raw) }
    }

    /// Returns the number of equalities in the given set.
    pub fn num_equalities(&self) -> isize {
        unsafe { mlirIntegerSetGetNumEqualities(self.raw) }
    }

    /// Returns the number of inequalities in the given set.
    pub fn num_inequalities(&self) -> isize {
        unsafe { mlirIntegerSetGetNumInequalities(self.raw) }
    }

    /// Returns the number of dimensions in the given set.
    pub fn num_dimensions(&self) -> isize {
        unsafe { mlirIntegerSetGetNumDims(self.raw) }
    }

    /// Returns the number of symbols in the given set.
    pub fn num_symbols(&self) -> isize {
        unsafe { mlirIntegerSetGetNumSymbols(self.raw) }
    }

    /// Returns the number of inputs (dimensions + symbols) in the given set.
    pub fn num_inputs(&self) -> isize {
        unsafe { mlirIntegerSetGetNumInputs(self.raw) }
    }

    /// Checks whether the given set is a canonical empty set, e.g., the set
    /// returned by mlirIntegerSetEmptyGet.
    pub fn is_empty(&self) -> bool {
        unsafe { mlirIntegerSetIsCanonicalEmpty(self.raw) }
    }

    /// Returns `pos`-th constraint of the set.
    pub fn get_constraint(&self, pos: isize) -> AffineExpr {
        unsafe { AffineExpr::from_raw(mlirIntegerSetGetConstraint(self.raw, pos)) }
    }

    /// Returns `true` of the `pos`-th constraint of the set is an equality
    /// constraint, `false` otherwise.
    pub fn is_constraint_eq(&self, pos: isize) -> bool {
        unsafe { mlirIntegerSetIsConstraintEq(self.raw, pos) }
    }
}

impl<'c> Eq for IntegerSet<'c> {}

impl<'c> PartialEq for IntegerSet<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirIntegerSetEqual(self.raw, other.to_raw()) }
    }
}
