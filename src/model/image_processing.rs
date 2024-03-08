use anyhow::Result;
use image::GenericImageView;
use ndarray::Array4;

pub fn check_input_image_size(image: &image::DynamicImage) -> Result<()> {
    let (width, height) = image.dimensions();
    if height != 400 || width % 200 != 0 {
        anyhow::bail!("Invalid input image size: {:?}", image.dimensions());
    }
    Ok(())
}

#[inline]
pub fn process_pair_classifier_ans_image(
    image: &mut image::DynamicImage,
    input_shape: (u32, u32),
    is_grayscale: bool,
) -> Result<Array4<f32>> {
    let image = crop_funcaptcha_ans_image(image);
    let sub_image = image.resize_exact(
        input_shape.0,
        input_shape.1,
        image::imageops::FilterType::Lanczos3,
    );

    // If the image is grayscale, we only need one channel
    image_processing(sub_image, input_shape, is_grayscale)
}

#[inline]
pub fn process_pair_classifier_image(
    image: &image::DynamicImage,
    index: (u32, u32),
    input_shape: (u32, u32),
    is_grayscale: bool,
) -> Result<Array4<f32>> {
    let (x, y) = (index.1 * 200, index.0 * 200);
    let sub_image = image.crop_imm(x, y, 200, 200).resize_exact(
        input_shape.0,
        input_shape.1,
        image::imageops::FilterType::Lanczos3,
    );

    // If the image is grayscale, we only need one channel
    image_processing(sub_image, input_shape, is_grayscale)
}

#[inline]
pub fn process_classifier_image(
    image: &mut image::DynamicImage,
    index: u32,
    input_shape: (u32, u32),
) -> Result<Array4<f32>> {
    let sub_image = crop_funcaptcha_image(image, (index / 3, index % 3), 100).resize_exact(
        input_shape.0,
        input_shape.1,
        image::imageops::FilterType::Lanczos3,
    );
    image_processing(sub_image, input_shape, false)
}

fn image_processing(
    sub_image: image::DynamicImage,
    input_shape: (u32, u32),
    is_grayscale: bool,
) -> Result<Array4<f32>> {
    if is_grayscale {
        let normalized_vec: Vec<f32> = sub_image
            .into_luma8()
            .into_raw()
            .into_iter()
            .map(|v| v as f32 / 255.0)
            .collect();
        let normalized_image = Array4::from_shape_vec(
            (1, 1, input_shape.0 as usize, input_shape.1 as usize),
            normalized_vec,
        )?;
        return Ok(normalized_image);
    }

    let normalized_vec: Vec<f32> = sub_image
        .into_rgb8()
        .into_raw()
        .into_iter()
        .map(|v| v as f32 / 255.0)
        .collect();
    let normalized_image = Array4::from_shape_vec(
        (1, input_shape.0 as usize, input_shape.1 as usize, 3),
        normalized_vec,
    )?;
    Ok(normalized_image.permuted_axes([0, 3, 1, 2]))
}

pub fn crop_funcaptcha_image(
    image: &mut image::DynamicImage,
    index: (u32, u32),
    width: u32,
) -> image::DynamicImage {
    let (x, y) = (index.1 * width, index.0 * width);
    image.crop_imm(x, y, width, width)
}

pub fn crop_funcaptcha_ans_image(image: &mut image::DynamicImage) -> image::DynamicImage {
    image.crop(0, 200, 135, 400)
}
