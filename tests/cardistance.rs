mod pre;
use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let dir = "tests/data/cardistance";
    let args = Default::default();
    let predictor = fs::onnx::new_predictor(Variant::Cardistance, &args)
        .await
        .unwrap();

    pre::test_predictor(dir, predictor);
}
