#![allow(dead_code)]

use anyhow::{anyhow, Error};
use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{One, Zero};

type Num = i32;
type Dim = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    elements: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Copy> Matrix<T> {
    pub fn from_vec(elements: Vec<T>, rows: usize, cols: usize) -> Result<Self, Error> {
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

    pub fn identity(n: usize) -> Self
    where
        T: One + Zero,
    {
        let mut res: Vec<T> = vec![Zero::zero(); n * n];

        let mut i;
        let mut j;
        for (idx, e) in res.iter_mut().enumerate() {
            i = index_to_coordinates(idx, n).0;
            j = index_to_coordinates(idx, n).1;
            if i == j {
                *e = One::one();
            }
        }
        Self::from_vec(res, n, n).unwrap()
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Add<Output = T>,
    {
        let (rows, cols) = self.dims_match(&rhs)?;
        Matrix::from_vec(
            self.iter()
                .flatten()
                .zip(rhs.iter().flatten())
                .map(|(&e1, &e2)| e1 + e2)
                .collect(),
            rows,
            cols,
        )
    }

    pub fn add_assign(&mut self, rhs: Self) -> Result<(), Error>
    where
        T: AddAssign,
    {
        self.dims_match(&rhs)?;
        self.iter_mut()
            .flatten()
            .zip(rhs.iter().flatten())
            .for_each(|(e1, &e2)| *e1 += e2);
        Ok(())
    }

    pub fn scalar_add(self, rhs: T) -> Result<Self, Error>
    where
        T: Add<Output = T>,
    {
        Matrix::from_vec(
            self.iter().flatten().map(|&e1| e1 + rhs).collect(),
            self.rows,
            self.cols,
        )
    }

    pub fn scalar_add_assign(&mut self, rhs: T)
    where
        T: AddAssign,
    {
        self.iter_mut().flatten().for_each(|e1| {
            *e1 += rhs;
        });
    }

    pub fn sub(self, rhs: Self) -> Result<Self, Error>
    where
        T: Sub<Output = T>,
    {
        let (rows, cols) = self.dims_match(&rhs)?;
        Matrix::from_vec(
            self.iter()
                .flatten()
                .zip(rhs.iter().flatten())
                .map(|(&e1, &e2)| e1 - e2)
                .collect(),
            rows,
            cols,
        )
    }

    pub fn sub_assign(&mut self, rhs: Self) -> Result<(), Error>
    where
        T: SubAssign,
    {
        self.dims_match(&rhs)?;
        self.iter_mut()
            .flatten()
            .zip(rhs.iter().flatten())
            .for_each(|(e1, &e2)| *e1 -= e2);
        Ok(())
    }

    pub fn scalar_sub(&self, rhs: T) -> Result<Self, Error>
    where
        T: Sub<Output = T>,
    {
        Matrix::from_vec(
            self.iter().flatten().map(|&e1| e1 - rhs).collect(),
            self.rows,
            self.cols,
        )
    }

    pub fn scalar_sub_assign(&mut self, rhs: T)
    where
        T: SubAssign,
    {
        self.iter_mut().flatten().for_each(|e1| {
            *e1 -= rhs;
        });
    }

    pub fn mul(self, rhs: Self) -> Result<Self, Error>
    where
        T: Mul<Output = T>,
    {
        todo!()
    }

    pub fn mul_assign(&mut self, rhs: Self) -> Result<(), Error>
    where
        T: SubAssign,
    {
        todo!()
    }

    pub fn scalar_mul(&self, rhs: T) -> Result<Self, Error>
    where
        T: Mul<Output = T>,
    {
        Matrix::from_vec(
            self.iter().flatten().map(|&e1| e1 * rhs).collect(),
            self.rows,
            self.cols,
        )
    }

    pub fn scalar_mul_assign(&mut self, rhs: T)
    where
        T: MulAssign,
    {
        self.iter_mut().flatten().for_each(|e1| {
            *e1 *= rhs;
        });
    }

    pub fn div(self, rhs: Self) -> Result<Self, Error>
    where
        T: Div<Output = T>,
    {
        todo!()
    }

    pub fn div_assign(&mut self, rhs: Self) -> Result<(), Error>
    where
        T: DivAssign,
    {
        todo!()
    }

    pub fn scalar_div(&self, rhs: T) -> Result<Self, Error>
    where
        T: Div<Output = T>,
    {
        Matrix::from_vec(
            self.iter().flatten().map(|&e1| e1 / rhs).collect(),
            self.rows,
            self.cols,
        )
    }

    pub fn scalar_div_assign(&mut self, rhs: T)
    where
        T: DivAssign,
    {
        self.iter_mut().flatten().for_each(|e1| {
            *e1 /= rhs;
        });
    }

    pub fn hadamard_product(&self, rhs: &Self) -> Result<Self, Error>
    where
        T: Mul<Output = T>,
    {
        let (rows, cols) = self.dims_match(&rhs)?;
        Matrix::from_vec(
            self.iter()
                .flatten()
                .zip(rhs.iter().flatten())
                .map(|(&x, &y)| x * y)
                .collect(),
            rows,
            cols,
        )
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

// Helper function to convert vector index to i and j coordinates
pub fn index_to_coordinates(idx: usize, row_length: usize) -> (usize, usize) {
    let i = idx / row_length + 1;
    let j = row_length - ((i * row_length) - idx) + 1;
    (i, j)
}

// Helper function to convert i and j to a vector index
pub fn coordinates_to_index(i: usize, j: usize, row_length: usize) -> usize {
    i * row_length + j
}

// Dot product helper for multiplication
pub fn dot_product(mat_a: &[Num], mat_b: &[Num], i: Dim, j: Dim, len: Dim) -> Num {
    let mut x;
    let mut y;
    let mut sum: Num = 0;
    // Normal dot product, multiply all numbers on a row of A and a column of B and keep a running sum
    for idx in 0..len {
        x = coordinates_to_index(idx, j, len);
        y = coordinates_to_index(i, idx, len);
        sum += mat_a[x] * mat_b[y];
    }
    sum
}

// Traditional matrix multiplication
pub fn matrix_multiplication(
    a: &[Num],
    b: &[Num],
    width_a: &Dim,
    depth_a: Dim,
    width_b: Dim,
    depth_b: &Dim,
) -> Result<Vec<Num>, Error> {
    let mut res: Vec<Num> = Vec::with_capacity(depth_a * width_b);

    // The width of the first input needs to match the depth of the second input
    if depth_b != width_a {
        Err(anyhow!("matrices must be identical dimensions"))
    } else {
        for rows in 0..depth_a {
            for cols in 0..width_b {
                res.push(dot_product(a, b, rows, cols, *width_a));
            }
        }
        Ok(res)
    }
}

// Matrix power
pub fn power(a: &[Num], width: &Dim, depth: Dim, pwr: Num) -> Vec<Num> {
    let mut res: Vec<Num> = Vec::with_capacity(depth * width);

    // First copy A into the result so it can be multiplied by itself (A * A)
    // This will help for further powers
    for idx in a {
        res.push(*idx);
    }

    // Now execute the power multiplication, will only execute once for squared
    let mut count = 1;
    loop {
        if count == pwr - 1 {
            break;
        }
        res = matrix_multiplication(&res, a, width, depth, *width, &depth).unwrap();
        count += 1;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition() -> Result<(), Error> {
        let e1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let e2 = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let e3 = vec![2, 4, 6, 8, 10, 12, 14, 16];

        let m1 = Matrix::from_vec(e1, 2, 4)?;
        let m2 = Matrix::from_vec(e2, 2, 4)?;

        let m3 = Matrix::from_vec(e3, 2, 4)?;

        assert_eq!(m1, m2);

        assert_eq!(m3, m1.add(&m2)?);

        Ok(())
    }

    #[test]
    fn identity() {
        let i = Matrix::<f64>::identity(5);
        dbg!(i);
    }

    #[test]
    fn transpose() {
        let m1_or_err = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2);
        let m2_or_err = Matrix::from_vec(vec![1, 3, 2, 4], 2, 2);

        assert!(m1_or_err.is_ok());
        assert!(m2_or_err.is_ok());

        let mut m1 = m1_or_err.unwrap();
        let m2 = m2_or_err.unwrap();

        m1.transpose();

        assert_eq!(m1, m2);
    }
}
