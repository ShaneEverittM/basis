mod math;

use neon::prelude::*;
use neon::result::Throw;

pub fn greeting(mut cx: FunctionContext) -> JsResult<JsString> {
    Ok(cx.string("Hello from basis"))
}

fn parse_number_arg(cx: &mut FunctionContext, idx: i32) -> Result<f64, Throw> {
    Ok(cx.argument::<JsNumber>(idx)?.value(cx))
}

pub fn add(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let x = parse_number_arg(&mut cx, 0)? as u64;
    let y = parse_number_arg(&mut cx, 1)? as u64;
    Ok(cx.number(math::add(x, y) as f64))
}

//Want to merely change the argument, think I did that right with a mutable reference?
//Scalar addition of a value x to the matrix
pub fn scalar_addition(v: &mut Vec<i32>, x: i32) {
    //For each element add x
    for i in v {
        *i += x;
    }
}

//Want to merely change the argument, think I did that right with a mutable reference?
//Scalar subtraction of a value x to the matrix
pub fn scalar_subtraction(v: &mut Vec<i32>, x: i32) {
    //For each element subtract x
    for i in v {
        *i -= x;
    }
}

//Want to merely change the argument, think I did that right with a mutable reference?
//Scalar multiplication of a value x to the matrix
pub fn scalar_multiplication(v: &mut Vec<i32>, x: i32) {
    //For each element multiply by x
    for i in  v {
        *i *= x;
    }
}

//TODO: Decide how we want to check the dimensions of the matrix? Could be as simple
//TODO: as passing the width and depth as parameters, which I did from here on
//Maybe pass as a struct with the values and the width and depth?

//Addition of two matrices A and B to produce C
pub fn matrix_addition(a: Vec<i32>, b: Vec<i32>, width_a: u32, depth_a: u32, width_b: u32, depth_b: u32) -> Vec<i32> {
    if (width_a != width_b) && (depth_a != depth_b) {
        //return mismatch error, ie cant do operation
    }
    //Can either make a new matrix or change one of the two, did both
    //Making a new matrix
    let mut c: Vec<i32> = Vec::new();
    let elements: usize = a.len();
    for num in 0 .. elements as u32 {
        c.push(&a[num as usize] + &b[num as usize]);
    }
    return c;

    //The other method, means changing way parameters are passed
    //Wasn't sure how to do a synchronized loop of both vectors so I just used the same usize iterator
    /*
    let elements: usize = a.len();
    for num in 0 .. elements as u32 {
        let old_bij = b[num as usize];
        b[num as usize] = a[num as usize] + old_bij;
    }
     */
}

//Subtraction of two matrices A and B to produce C
pub fn matrix_subtraction(a: Vec<i32>, b: Vec<i32>, width_a: u32, depth_a: u32, width_b: u32, depth_b: u32) -> Vec<i32> {
    if (width_a != width_b) && (depth_a != depth_b) {
        //return mismatch error
    }
    //Can either make a new matrix or change one of the two, did both
    //Making a new matrix
    let mut c: Vec<i32> = Vec::new();
    let elements: usize = a.len();
    for num in 0 .. elements as u32 {
        c.push(&a[num as usize] - &b[num as usize]);
    }
    return c;

    //The other method, means changing way parameters are passed
    //Wasn't sure how to do a synchronized loop of both vectors so I just used the same usize iterator
    /*
    let elements: usize = a.len();
    for num in 0 .. elements as u32 {
        let old_bij = b[num as usize];
        b[num as usize] = a[num as usize] - old_bij;
    }
     */
}

//Hadamard product of two matrices A and B to produce C
pub fn hadamard_product(a: Vec<i32>, b: Vec<i32>, width_a: u32, depth_a: u32, width_b: u32, depth_b: u32)
    -> Vec<i32> {
    if (width_a != width_b) && (depth_a != depth_b) {
        //return mismatch error
    }
    //Can either make a new matrix or change one of the two, did both
    //Making a new matrix
    let mut c: Vec<i32> = Vec::new();
    let elements: usize = a.len();
    for num in 0 .. elements as u32 {
        c.push(&a[num as usize] * &b[num as usize]);
    }
    return c;

    //The other method, means changing way parameters are passed
    //Wasn't sure how to do a synchronized loop of both vectors so I just used the same usize iterator
    /*
    let elements: usize = a.len();
    for num in 0 .. elements as u32 {
        let old_bij = b[num as usize];
        b[num as usize] = a[num as usize] * old_bij;
    }
     */
}

//Helper function to convert vector index to i and j coordinates
//Want to return two u32s i and j, use a tuple?
pub fn index_to_coordinates(idx: u32, d: u32) -> (u32, u32) {
    let i = idx / d + 1;
    let j = idx - i;
    return (i, j);
}

//Helper function to convert i and j to a vector index
pub fn coordinates_to_index(i: u32, j: u32) -> u32 {
    return i * j - 1;
}

//Transpose a matrix
pub fn transpose(v: Vec<i32>, width: u32, depth: u32) -> Vec<i32>{
    //Allocate new vector of the same size as the input, correct?
    let mut res: Vec<i32> = Vec::with_capacity((width * depth) as usize);

    //Used to iterate
    let elements: usize = v.len();
    for num in 0 .. elements as u32 {
        //Get the i and j coordinate values from the current index
        let i = index_to_coordinates(num, depth).0;    //Shadowing ok here?
        let j = index_to_coordinates(num, depth).1;    //Shadowing ok here?

        //Transpose by flipping i and j
        let trans_idx = coordinates_to_index(j, i);

        //Push the corresponding value to the new transposed matrix
        res.push(v[trans_idx as usize]);
    }
    return res;
}

#[neon::main]
fn main(mut m: ModuleContext) -> NeonResult<()> {
    m.export_function("greeting", greeting)?;
    m.export_function("add", add)?;
    Ok(())
}