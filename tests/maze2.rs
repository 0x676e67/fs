mod pre;
use fs::onnx::Variant;

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test() {
    let dir = "tests/data/maze2";
    let args = Default::default();
    let predictor = fs::onnx::new_predictor(Variant::Maze2, &args)
        .await
        .unwrap();

    pre::test_predictor(dir, predictor).unwrap();
}
