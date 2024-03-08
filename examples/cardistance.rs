use fcsrv::{model::ModelType, BootArgs};
use ort::AllocatorType;
use std::path::PathBuf;

fn main() {
    fcsrv::model::init_predictor(&BootArgs {
        debug: false,
        bind: "0.0.0.0:8000".parse().unwrap(),
        tls_cert: None,
        tls_key: None,
        api_key: None,
        multi_image_limit: 1,
        update_check: false,
        model_dir: Some(PathBuf::from("models")),
        num_threads: 4,
        allocator: AllocatorType::Arena,
        fallback_solver: None,
        fallback_key: None,
        fallback_developer_id: None,
        fallback_endpoint: None,
    })
    .unwrap();

    let predictor = fcsrv::model::get_predictor(ModelType::Cardistance).unwrap();

    let image_file = std::fs::read(
        "images/cardistance/0a969ea283f3d76599e4a5d85aa8ca3db4dbf45ac8166ba8fe5915533e2c149b.jpg",
    )
    .unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 7);
}
