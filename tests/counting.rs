use fs::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();

    let predictor = fs::onnx::get_predictor(Variant::Counting, &args)
        .await
        .unwrap();

    let image_file =
        std::fs::read("counting/0a1d5e94-8187-4124-a999-3ab7af6cb5e3.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 0);
}
