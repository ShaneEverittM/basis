use neon::prelude::*;

pub fn greeting(mut cx: FunctionContext) -> JsResult<JsString> {
    Ok(cx.string("Hello from basis"))
}

#[neon::main]
fn main(mut m: ModuleContext) -> NeonResult<()> {
    m.export_function("greeting", greeting)
}