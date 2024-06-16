use fcsrv::onnx::Variant;

#[tokio::main]
async fn main() {
    let args = Default::default();
    let predictor = fcsrv::onnx::get_predictor(Variant::Shadows, &args)
        .await
        .unwrap();

    // Read image file images/shadows/0bf82e3c5abec9553e21c8a8515d7b6f3d94545eff465d9f5ff4e23fb07b0741_1.jpg
    let image_file = std::fs::read(
        "images/shadows/0bf82e3c5abec9553e21c8a8515d7b6f3d94545eff465d9f5ff4e23fb07b0741_1.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 1);

    // Read image file images/shadows/0d1dd3dcfa12b88027135334db1b08a824adfbc0688200324d935043e121e7b7_3.jpg
    let image_file = std::fs::read(
        "images/shadows/0d1dd3dcfa12b88027135334db1b08a824adfbc0688200324d935043e121e7b7_3.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 3);

    // Read image file images/shadows/1d5e432bffabb5d6a32cf06381d43003c2d1f4ad380ffe464a6ae7cf60db4e74_2.jpg
    let image_file = std::fs::read(
        "images/shadows/1d5e432bffabb5d6a32cf06381d43003c2d1f4ad380ffe464a6ae7cf60db4e74_2.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 2);

    // Read image file images/shadows/1ee9fb5afa79bcc27c9f5e01b2e995b7db9fb6fae41b97912f7a5df6f3bf7d14_1.jpg
    let image_file = std::fs::read(
        "images/shadows/1ee9fb5afa79bcc27c9f5e01b2e995b7db9fb6fae41b97912f7a5df6f3bf7d14_1.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 1);

    // Read image file images/shadows/2d3d456cf6938f721685d73d94bc00ff511fd23462c0a55ab897dae0d3617e94_0.jpg
    let image_file = std::fs::read(
        "images/shadows/2d3d456cf6938f721685d73d94bc00ff511fd23462c0a55ab897dae0d3617e94_0.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 0);
}
