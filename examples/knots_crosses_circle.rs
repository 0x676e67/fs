use fc::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();

    let predictor = fc::onnx::get_predictor(Variant::KnotsCrossesCircle, &args)
        .await
        .unwrap();

    let image_file = std::fs::read("docs/knotsCrossesCircle/knotsCrossesCircle_0.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4);
}
