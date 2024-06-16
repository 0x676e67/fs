use fc::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();
    let predictor = fc::onnx::get_predictor(Variant::M3dRollballAnimals, &args)
        .await
        .unwrap();

    // Read image file docs/3d_rollball_animals/0bcc74b7-487c-4db4-8d48-7d2d2091ae23_3.jpg
    let image_file =
        std::fs::read("docs/3d_rollball_animals/0bcc74b7-487c-4db4-8d48-7d2d2091ae23_3.jpg")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 3);

    // Read image file docs/3d_rollball_animals/1a03913c-61e1-4c95-a9c6-e45bbc419ee4-0_3.jpg
    let image_file =
        std::fs::read("docs/3d_rollball_animals/1a03913c-61e1-4c95-a9c6-e45bbc419ee4-0_3.jpg")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 3);

    // Read image file docs/3d_rollball_animals/1a03913c-61e1-4c95-a9c6-e45bbc419ee4-1_3.jpg
    let image_file =
        std::fs::read("docs/3d_rollball_animals/1a03913c-61e1-4c95-a9c6-e45bbc419ee4-1_3.jpg")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 3);

    // Read image file docs/3d_rollball_animals/1a03913c-61e1-4c95-a9c6-e45bbc419ee4-2_2.jpg
    let image_file =
        std::fs::read("docs/3d_rollball_animals/1a03913c-61e1-4c95-a9c6-e45bbc419ee4-2_2.jpg")
            .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 2);
}
