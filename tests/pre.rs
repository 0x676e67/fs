use fs::onnx::Predictor;
use std::sync::Arc;

pub fn test_predictor(dir: &str, predictor: Arc<dyn Predictor>) {
    // list all files in the directory
    let files = std::fs::read_dir(dir).unwrap();
    for file in files {
        let file = file.unwrap();
        // get .jpg files
        let filepath = file.file_name();
        let filepath = filepath.to_string_lossy();
        if filepath.ends_with(".jpg") {
            let image_file = std::fs::read(file.path()).unwrap();
            let guess = predictor
                .predict(image::load_from_memory(&image_file).unwrap())
                .unwrap();

            // read the label from the filename.txt
            let label_file = std::fs::read(file.path().with_extension("txt")).unwrap();
            let label = {
                let label = String::from_utf8(label_file).unwrap();
                let label = label.trim();
                label.parse::<i32>().unwrap()
            };

            assert_eq!(guess, label);
        }
    }
}
