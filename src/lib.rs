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
    let x = parse_number_arg(&mut cx, 0)?;
    let y = parse_number_arg(&mut cx, 1)?;
    Ok(cx.number(math::add(x, y)))
}

#[neon::main]
fn main(mut m: ModuleContext) -> NeonResult<()> {
    m.export_function("greeting", greeting)?;
    m.export_function("add", add)?;
    Ok(())
}