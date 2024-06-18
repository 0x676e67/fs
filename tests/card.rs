use fs::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();

    let predictor = fs::onnx::get_predictor(Variant::Card, &args).await.unwrap();

    let image_file = std::fs::read("card/card_1.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 0);

    let image_file = std::fs::read("card/card_2.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 2);
}
