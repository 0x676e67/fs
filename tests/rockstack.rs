use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let args = Default::default();

    let predictor = fs::onnx::new_predictor(Variant::Rockstack, &args)
        .await
        .unwrap();

    let image_file = std::fs::read("tests/data/rockstack/9258513.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 3);

    let image_file = std::fs::read("tests/data/rockstack/50444558.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 2);
}
