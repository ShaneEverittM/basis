#![allow(dead_code)]

use anyhow::{anyhow, Error};
use std::ops::{Add, Deref, DerefMut, Index};

type Num = i32;
type Dim = usize;

#[derive(Debug)]
struct Matrix<T: Sized + Copy + Default, const R: usize, const C: usize> {
    elements: [[T; C]; R],
}

impl<T: Sized + Copy + Default, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn new(mut elements: Vec<T>) -> Self {
        let mut rows = [[T::default(); C]; R];
        for (i, slice) in elements.chunks_mut(C).enumerate() {
            let mut x = [T::default(); C];
            for (i, e) in slice.iter_mut().enumerate() {
                x[i] = *e;
            }
            rows[i] = x;
        }

        Self { elements: rows }
    }

    pub fn new_unsafe(elements: Vec<T>) -> Self {
        Self {
            elements: unsafe { *(elements.as_ptr() as *mut [[T; C]; R]) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_test() {
        let elements = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let m = unsafe { Matrix::<i32, 3, 3>::new_unsafe(elements) };
        dbg!(m);
    }
}

//
// impl Matrix {
//     pub fn from_vec(cols: usize, rows: usize, elements: Vec<i32>) -> Self {
//         Self {
//             cols,
//             rows,
//             elements,
//         }
//     }
// }

// impl Deref for Matrix {
//     type Target = Vec<i32>;
//
//     fn deref(&self) -> &Self::Target {
//         &self.elements
//     }
// }
//
// impl DerefMut for Matrix {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.elements
//     }
// }
//
// impl Index<[usize; 2]> for Matrix {
//     type Output = i32;
//
//     fn index(&self, index: [usize; 2]) -> &Self::Output {
//         &self.elements[index[0] * self.cols + index[1]]
//     }
// }
//
// impl Add<Matrix> for Matrix {
//     type Output = Matrix;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         let elements = self.iter().zip(rhs.iter()).map(|(x, y)| x + y).collect();
//         Matrix::from_vec(self.rows, self.cols, elements)
//     }
// }
//
// impl FromIterator<i32> for Matrix {
//     fn from_iter<T: IntoIterator<Item = i32>>(iter: T) -> Self {
//         let e: Vec<i32> = iter.into_iter().collect();
//         Self::from_vec(0, 0, e)
//     }
// }

// impl Add<i32> for Matrix {
//     type Output = Matrix;
//
//     fn add(self, rhs: i32) -> Self::Output {
//     }
// }

// Want to merely change the argument, think I did that right with a mutable reference?
// Scalar addition of a value x to the matrix
pub fn scalar_addition(v: &mut Vec<Num>, x: Num) {
    // For each element add x
    for i in v {
        *i += x;
    }
}

// Want to merely change the argument, think I did that right with a mutable reference?
// Scalar subtraction of a value x to the matrix
pub fn scalar_subtraction(v: &mut Vec<Num>, x: Num) {
    // For each element subtract x
    for i in v {
        *i -= x;
    }
}

// Want to merely change the argument, think I did that right with a mutable reference?
// Scalar multiplication of a value x to the matrix
pub fn scalar_multiplication(v: &mut Vec<Num>, x: Num) {
    // For each element multiply by x
    for i in v {
        *i *= x;
    }
}

// TODO: Decide how we want to check the dimensions of the matrix? Could be as simple
// TODO: as passing the width and depth as parameters, which I did from here on
// Maybe pass as a struct with the values and the width and depth?

// Addition of two matrices A and B to produce C
pub fn matrix_addition(
    a: Vec<Num>,
    b: Vec<Num>,
    width_a: Dim,
    depth_a: Dim,
    width_b: Dim,
    depth_b: Dim,
) -> Result<Vec<Num>, Error> {
    if (width_a != width_b) || (depth_a != depth_b) {
        Err(anyhow!("matrices must be identical dimensions"))
    } else {
        Ok(a.into_iter()
            .zip(b.into_iter())
            .map(|(x, y)| x + y)
            .collect())
    }
}

// Subtraction of two matrices A and B to produce C
pub fn matrix_subtraction(
    a: Vec<Num>,
    b: Vec<Num>,
    width_a: Dim,
    depth_a: Dim,
    width_b: Dim,
    depth_b: Dim,
) -> Result<Vec<Num>, Error> {
    if (width_a != width_b) || (depth_a != depth_b) {
        Err(anyhow!("matrices must be identical dimensions"))
    } else {
        Ok(a.into_iter()
            .zip(b.into_iter())
            .map(|(x, y)| x - y)
            .collect())
    }
}

// Hadamard product of two matrices A and B to produce C
pub fn hadamard_product(
    a: Vec<Num>,
    b: Vec<Num>,
    width_a: Dim,
    depth_a: Dim,
    width_b: Dim,
    depth_b: Dim,
) -> Result<Vec<Num>, Error> {
    if (width_a != width_b) || (depth_a != depth_b) {
        Err(anyhow!("matrices must be identical dimensions"))
    } else {
        Ok(a.into_iter()
            .zip(b.into_iter())
            .map(|(x, y)| x * y)
            .collect())
    }
}

// Helper function to convert vector index to i and j coordinates
pub fn index_to_coordinates(idx: usize, d: usize) -> (usize, usize) {
    let i = idx / d + 1;
    let j = d - ((i * d) - idx) + 1;
    (i, j)
}

// Helper function to convert i and j to a vector index
pub fn coordinates_to_index(i: usize, j: usize) -> usize {
    i * j - 1
}

// Transpose a matrix
pub fn transpose(v: Vec<Num>, width: Dim, depth: Dim) -> Vec<Num> {
    // Allocate new vector of the same size as the input, correct?
    let mut res: Vec<Num> = Vec::with_capacity(width * depth);

    // Used to iterate
    let mut i;
    let mut j;
    for idx in 0..v.len() {
        // Get the i and j coordinate values from the current index
        i = index_to_coordinates(idx, depth).0;
        j = index_to_coordinates(idx, depth).1;

        // Transpose by flipping i and j
        let trans_idx = coordinates_to_index(j, i);

        // Push the corresponding value to the new transposed matrix
        res.push(v[trans_idx]);
    }
    res
}

// Dot product helper for multiplication
pub fn dot_product(mat_a: &[Num], mat_b: &[Num], i: Dim, j: Dim, len: &Dim) -> Num {
    let mut x;
    let mut y;
    let mut sum: Num = 0;
    // Normal dot product, multiply all numbers on a row of A and a column of B and keep a running sum
    for idx in 0..*len {
        x = coordinates_to_index(idx, j);
        y = coordinates_to_index(i, idx);
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
                res.push(dot_product(a, b, rows, cols, width_a));
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

// Generate an identity matrix for the provided matrix dimensions
pub fn generate_identity(n: Dim) -> Vec<Num> {
    let mut res: Vec<Num> = Vec::with_capacity(n * n);

    let mut i;
    let mut j;
    let size = n * n;
    for idx in 0..size {
        i = index_to_coordinates(idx, n).0;
        j = index_to_coordinates(idx, n).1;
        if i == j {
            res.push(1);
        } else {
            res.push(0);
        }
    }
    res
}
