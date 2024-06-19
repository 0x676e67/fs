use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let args = Default::default();

    let predictor = fs::onnx::get_predictor(Variant::Frankenhead, &args)
        .await
        .unwrap();

    let image_file =
        std::fs::read("tests/data/frankenhead/0a645367c6d7857122a66b43e9cb6e1d.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4);
}
