mod pre;
use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let dir = "tests/data/hand_number_puzzle";
    let args = Default::default();
    let predictor = fs::onnx::new_predictor(Variant::HandNumberPuzzle, &args)
        .await
        .unwrap();

    pre::test_predictor(dir, predictor);
}
