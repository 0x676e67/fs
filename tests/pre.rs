use fs::onnx::Predictor;
use std::{error::Error, sync::Arc};

pub fn test_predictor(dir: &str, predictor: Arc<dyn Predictor>) -> Result<(), Box<dyn Error>> {
    // list all files in the directory
    for file in std::fs::read_dir(dir)? {
        let file = file?;
        // get .jpg files
        let filepath = file.file_name();
        let filepath = filepath.to_string_lossy();
        if filepath.ends_with(".jpg") {
            let image_file = std::fs::read(file.path())?;
            let guess = predictor.predict(image::load_from_memory(&image_file)?)?;

            // read the label from the filename.txt
            let label = {
                let lable_file = file.path().with_extension("txt");
                let parse_lable = || -> Result<i32, Box<dyn Error>> {
                    let label_file = std::fs::read(&lable_file)?;
                    let label = String::from_utf8(label_file)?;
                    let label = label.trim();
                    label.parse::<i32>().map_err(|e| e.into())
                };

                if lable_file.exists() {
                    parse_lable()?
                } else {
                    // parse the label from the filename 2024-06-28-09-42-1321773_marked_3.jpg
                    let label = filepath
                        .split('_')
                        .last()
                        .unwrap()
                        .split('.')
                        .next()
                        .unwrap();
                    label.parse::<i32>()?
                }
            };

            assert_eq!(guess, label);
            println!("{}: {}", filepath, guess);
        }
    }

    Ok(())
}
