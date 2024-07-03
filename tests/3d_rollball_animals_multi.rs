mod pre;
use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let dir = "tests/data/3d_rollball_animals_multi";
    let args = Default::default();
    let predictor = fs::onnx::new_predictor(Variant::RollballAnimalsMulti, &args)
        .await
        .unwrap();

    pre::test_predictor(dir, predictor);
}
