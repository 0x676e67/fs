use fs::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();
    let predictor = fs::onnx::get_predictor(Variant::PenguinsIcon, &args)
        .await
        .unwrap();

    let image_file = std::fs::read(
        "penguins-icon/0a36f4aedb149bd1aa28f26094799253f7c8228ae0cf49c4c72f6a4e76b2782f.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4)
}
