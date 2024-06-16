use fcsrv::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();

    let predictor = fcsrv::onnx::get_predictor(Variant::HandNumberPuzzle, &args)
        .await
        .unwrap();

    let image_file = std::fs::read("images/hand_number_puzzle/hand_number_puzzle_0.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 4);

    let image_file = std::fs::read("images/hand_number_puzzle/hand_number_puzzle_1.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 5);
}
