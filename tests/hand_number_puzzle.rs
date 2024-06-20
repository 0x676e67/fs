use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let args = Default::default();

    let predictor = fs::onnx::new_predictor(Variant::HandNumberPuzzle, &args)
        .await
        .unwrap();

    let image_file =
        std::fs::read("tests/data/hand_number_puzzle/hand_number_puzzle_0.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4);

    let image_file =
        std::fs::read("tests/data/hand_number_puzzle/hand_number_puzzle_1.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 5);
}
