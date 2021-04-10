mod matrix;

use neon::prelude::*;
use neon::result::Throw;

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
    Ok(cx.number((x + y) as f64))
}

#[neon::main]
fn main(mut m: ModuleContext) -> NeonResult<()> {
    m.export_function("greeting", greeting)?;
    m.export_function("add", add)?;
    Ok(())
}
