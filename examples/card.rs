use fc::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();

    let predictor = fc::onnx::get_predictor(Variant::Card, &args)
        .await
        .unwrap();

    let image_file = std::fs::read("docs/card/card_1.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 0);

    let image_file = std::fs::read("docs/card/card_2.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 2);
}
