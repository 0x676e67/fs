use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let args = Default::default();
    let predictor = fs::onnx::new_predictor(Variant::PenguinsIcon, &args)
        .await
        .unwrap();

    let image_file = std::fs::read(
        "tests/data/penguins-icon/0a36f4aedb149bd1aa28f26094799253f7c8228ae0cf49c4c72f6a4e76b2782f.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4)
}
