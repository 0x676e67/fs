use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn my_test() {
    let args = Default::default();

    let predictor = fs::onnx::new_predictor(Variant::LumberLengthGame, &args)
        .await
        .unwrap();

    let image_file =
        std::fs::read("tests/data/lumber-length-game/000b4ffb7c926764149115beea219279.png")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 3);

    let image_file =
        std::fs::read("tests/data/lumber-length-game/000b08cbd177289a296d04f43c459661.png")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 1);
}
