use fs::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();

    let predictor = fs::onnx::get_predictor(Variant::BrokenJigsawbrokenjigsaw_swap, &args)
        .await
        .unwrap();

    let image_file =
        std::fs::read("docs/BrokenJigsawbrokenjigsaw_swap/1_0763dbd0d732075bba1fda9da7046f10.jpg")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 1);
}
