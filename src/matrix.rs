#![allow(dead_code)]

use anyhow::{anyhow, Error};
use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{Bounded, One, Zero};
use std::convert::{TryFrom, TryInto};
use std::iter::Sum;

type Num = i32;
type Dim = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    elements: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Copy> Matrix<T> {
    pub fn from_flat_vec(elements: Vec<T>, rows: usize, cols: usize) -> Result<Self, Error> {
        if rows * cols != elements.len() {
            return Err(anyhow!("Rows * columns must equal length of elements vec"));
        }

        let mut v: VecDeque<T> = elements.into();
        let mut acc: Vec<Vec<T>> = Vec::new();

        while v.len() >= cols {
            acc.push(v.drain(0..cols).collect());
            v.shrink_to_fit();
        }

        if !v.is_empty() {
            return Err(anyhow!("Must subdivide into rows"));
        }

        Ok(Self {
            elements: acc,
            rows,
            cols,
        })
    }

    pub fn from_array<const R: usize, const C: usize>(slice: [[T; C]; R]) -> Self {
        let elements: Vec<Vec<T>> = Vec::from(slice)
            .iter_mut()
            .map(|&mut row| Vec::from(row))
            .collect();

        Self {
            elements,
            rows: R,
            cols: C,
        }
    }

    pub fn from_vec(elements: Vec<Vec<T>>) -> Result<Self, Error> {
        let rows = elements.len();
        let cols = elements.first().unwrap().len();
        if elements.iter().any(|r| r.len() != cols) {
            Err(anyhow!("Bumpy matrix"))
        } else {
            Ok(Self {
                elements,
                rows,
                cols,
            })
        }
    }

    pub fn identity(n: usize) -> Self
    where
        T: One + Zero,
    {
        let mut res: Vec<T> = vec![Zero::zero(); n * n];

        let mut i;
        let mut j;
        for (idx, e) in res.iter_mut().enumerate() {
            i = Matrix::<T>::index_to_coordinates(idx, n).0;
            j = Matrix::<T>::index_to_coordinates(idx, n).1;
            if i == j {
                *e = One::one();
            }
        }
        Self::from_flat_vec(res, n, n).unwrap()
    }

    pub fn rows(&self) -> impl Iterator<Item = &Vec<T>> + '_ {
        self.elements.iter()
    }

    pub fn cols(&self) -> impl Iterator<Item = &Vec<T>> + '_ {
        let mut transpose = self.clone();
        transpose.transpose();
        self.elements.iter()
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Add<Output = T>,
    {
        self.element_wise_arith_op(rhs, Add::add)
    }

    pub fn add_assign(&mut self, rhs: &Self) -> Result<(), Error>
    where
        T: AddAssign,
    {
        self.element_wise_update_op(rhs, AddAssign::add_assign)
    }

    pub fn scalar_add(&self, rhs: T) -> Self
    where
        T: Add<Output = T>,
    {
        self.scalar_op(rhs, Add::add)
    }

    pub fn scalar_add_assign(&mut self, rhs: T)
    where
        T: AddAssign,
    {
        self.iter_mut().flatten().for_each(|e1| {
            *e1 += rhs;
        });
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Sub<Output = T>,
    {
        self.element_wise_arith_op(rhs, std::ops::Sub::sub)
    }

    pub fn sub_assign(&mut self, rhs: &Self) -> Result<(), Error>
    where
        T: SubAssign,
    {
        self.element_wise_update_op(rhs, SubAssign::sub_assign)
    }

    pub fn scalar_sub(&self, rhs: T) -> Self
    where
        T: Sub<Output = T>,
    {
        self.scalar_op(rhs, Sub::sub)
    }

    pub fn scalar_sub_assign(&mut self, rhs: T)
    where
        T: SubAssign,
    {
        self.scalar_update_op(rhs, SubAssign::sub_assign)
    }

    pub fn scalar_mul(&self, rhs: T) -> Self
    where
        T: Mul<Output = T>,
    {
        self.scalar_op(rhs, Mul::mul)
    }

    pub fn scalar_mul_assign(&mut self, rhs: T)
    where
        T: MulAssign,
    {
        self.scalar_update_op(rhs, MulAssign::mul_assign)
    }

    pub fn scalar_div(&self, rhs: T) -> Self
    where
        T: Div<Output = T>,
    {
        self.scalar_op(rhs, Div::div)
    }

    pub fn scalar_div_assign(&mut self, rhs: T)
    where
        T: DivAssign,
    {
        self.scalar_update_op(rhs, DivAssign::div_assign)
    }

    pub fn hadamard_product(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Mul<Output = T>,
    {
        self.element_wise_arith_op(rhs, Mul::mul)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Mul<Output = T> + Sum,
    {
        let mut result = Vec::new();
        for row in self.rows() {
            let mut row_result: Vec<T> = Vec::new();
            for col in rhs.cols() {
                let new_row = row.iter().zip(col.iter()).map(|(&x, &y)| x * y).sum();
                row_result.push(new_row);
            }
            result.push(row_result)
        }
        Matrix::try_from(result)
    }

    pub fn mul_assign(&mut self, rhs: &Self) -> Result<(), Error>
    where
        T: Mul<Output = T> + Sum,
    {
        // Have to create a copy here because in place multiplication is impossible
        *self = self.clone().mul(&rhs)?;
        Ok(())
    }

    pub fn exp(&self, power: isize) -> Result<Self, Error>
    where
        T: Mul<Output = T> + Sum + Zero + One + Bounded,
    {
        // Cannot use ranges in match on generics
        #[allow(clippy::comparison_chain)]
        if power > 0 {
            let mut base = self.clone();

            for _ in 0..power {
                base.mul_assign(&self)?;
            }

            Ok(base)
        } else if power < 0 {
            let mut inverse = self.clone();
            inverse.invert();
            inverse.exp(power.abs())
        } else {
            Ok(Self::identity(self.rows))
        }
    }

    pub fn transpose(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if j < i {
                    // SAFETY x and y are aligned, non-null and non-overlapping.
                    // The swap will incur no re-allocations or moves so Vec will not be
                    // disturbed.
                    unsafe {
                        std::ptr::swap_nonoverlapping(
                            &mut self[i][j] as *mut T,
                            &mut self[j][i] as *mut T,
                            1,
                        );
                    }
                }
            }
        }
    }

    pub fn invert(&mut self) {
        todo!()
    }

    fn element_wise_arith_op(&self, rhs: &Self, op: impl Fn(T, T) -> T) -> Result<Self, Error> {
        self.dims_match(&rhs)?;

        self.iter()
            .zip(rhs.iter())
            .map(|(row1, row2)| {
                row1.iter()
                    .zip(row2.iter())
                    .map(|(&x, &y)| op(x, y))
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .try_into()
    }

    fn element_wise_update_op(
        &mut self,
        rhs: &Self,
        mut op: impl FnMut(&mut T, T),
    ) -> Result<(), Error> {
        self.dims_match(&rhs)?;

        self.iter_mut()
            .flatten()
            .zip(rhs.iter().flatten())
            .for_each(|(x, &y)| op(x, y));

        Ok(())
    }

    fn scalar_op(&self, rhs: T, op: impl Fn(T, T) -> T) -> Self {
        Self {
            elements: self
                .iter()
                .map(|row| row.iter().map(|&e| op(e, rhs)).collect::<Vec<T>>())
                .collect::<Vec<Vec<T>>>(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn scalar_update_op(&mut self, rhs: T, mut op: impl FnMut(&mut T, T)) {
        self.iter_mut().flatten().for_each(|e| op(e, rhs))
    }

    fn dims_match(&self, other: &Self) -> Result<(usize, usize), Error> {
        if self.rows == other.rows && self.cols == other.cols {
            Ok((self.rows, self.cols))
        } else {
            Err(anyhow!("Dimension mismatch"))
        }
    }

    fn index_to_coordinates(idx: usize, row_length: usize) -> (usize, usize) {
        let i = idx / row_length + 1;
        let j = row_length - ((i * row_length) - idx) + 1;
        (i, j)
    }

    fn coordinates_to_index(i: usize, j: usize, row_length: usize) -> usize {
        i * row_length + j
    }
}

impl<T: Copy + Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in self.iter() {
            for e in row.iter() {
                f.write_fmt(format_args!("{}, ", e))?;
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

impl<T: Copy> Eq for Matrix<T> where T: Eq {}

impl<T: Copy> Deref for Matrix<T> {
    type Target = Vec<Vec<T>>;

    fn deref(&self) -> &Self::Target {
        &self.elements
    }
}

impl<T: Copy> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.elements
    }
}
impl<T: Copy, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(array: [[T; C]; R]) -> Self {
        Self::from_array(array)
    }
}

impl<T: Copy> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = Error;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        Self::from_vec(value)
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    macro_rules! ok {
        ($m:expr) => {{
            let m = $m;
            assert!(m.is_ok());
            m.unwrap()
        }};
    }

    #[test]
    fn addition() {
        let mut A = Matrix::from([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let B = Matrix::from([[8, 7, 6, 5], [4, 3, 2, 1]]);

        let C = Matrix::from([[9, 9, 9, 9], [9, 9, 9, 9]]);

        assert_eq!(C, ok!(A.add(&B)));

        ok!(A.add_assign(&B));

        assert_eq!(A, C);

        let scalar = 1;

        let E = Matrix::from([[10, 10, 10, 10], [10, 10, 10, 10]]);

        let D = C.scalar_add(scalar);

        assert_eq!(D, E);
    }

    #[test]
    fn identity() {
        let identity = Matrix::identity(5);
        let manual_identity = Matrix::from([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]);

        assert_eq!(manual_identity, identity);
    }

    #[test]
    fn transpose() {
        let mut m1 = Matrix::from([[1, 2], [3, 4]]);
        let m2 = Matrix::from([[1, 3], [2, 4]]);

        m1.transpose();

        assert_eq!(m1, m2);
    }

    #[test]
    fn multiply() {
        let m1 = Matrix::from([[1, 2], [3, 4]]);
        let m2 = Matrix::from([[1, 3], [2, 4]]);

        let m3 = ok!(m1.mul(&m2));

        dbg!(m3);
    }

    #[test]
    fn from_vector() {
        let m1 = ok!(Matrix::try_from(vec![vec![1, 2], vec![3, 4]]));
        let m2 = Matrix::from([[1, 2], [3, 4]]);

        assert_eq!(m1, m2);
    }
}
