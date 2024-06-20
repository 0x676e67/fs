use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let args = Default::default();

    let predictor = fs::onnx::new_predictor(Variant::KnotsCrossesCircle, &args)
        .await
        .unwrap();

    let image_file =
        std::fs::read("tests/data/knotsCrossesCircle/knotsCrossesCircle_0.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4);
}
