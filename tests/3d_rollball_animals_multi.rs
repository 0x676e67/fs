use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let args = Default::default();
    let predictor = fs::onnx::new_predictor(Variant::RollballAnimalsMulti, &args)
        .await
        .unwrap();

    // Read image file
    // 3d_rollball_animals/0bcc74b7-487c-4db4-8d48-7d2d2091ae23_3.jpg
    let image_file =
        std::fs::read("tests/data/3d_rollball_animals_multi/0a0f51e3357fe28527fda31b51757915.png")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 6);
}
