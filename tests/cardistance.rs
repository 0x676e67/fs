use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let args = Default::default();

    let predictor = fs::onnx::new_predictor(Variant::Cardistance, &args)
        .await
        .unwrap();

    let image_file = std::fs::read(
        "tests/data/cardistance/0a969ea283f3d76599e4a5d85aa8ca3db4dbf45ac8166ba8fe5915533e2c149b.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 7);
}
