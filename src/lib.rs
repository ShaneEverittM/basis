mod math;

use neon::prelude::*;
use neon::result::Throw;

use anyhow::{anyhow, Error};

type Num = i32;
type Dim = usize;

pub fn greeting(mut cx: FunctionContext) -> JsResult<JsString> {
    Ok(cx.string("Hello from basis"))
}

fn parse_number_arg(cx: &mut FunctionContext, idx: i32) -> Result<f64, Throw> {
    Ok(cx.argument::<JsNumber>(idx)?.value(cx))
}

//  Dumb example
pub fn add(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let x = parse_number_arg(&mut cx, 0)? as u64;
    let y = parse_number_arg(&mut cx, 1)? as u64;
    Ok(cx.number(math::add(x, y) as f64))
}

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
    let j = idx - i;
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

#[neon::main]
fn main(mut m: ModuleContext) -> NeonResult<()> {
    m.export_function("greeting", greeting)?;
    m.export_function("add", add)?;
    Ok(())
}
