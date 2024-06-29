use std::ops::{Deref, DerefMut};

use indicatif::ProgressStyle;

pub struct ProgressBar(indicatif::ProgressBar);

impl ProgressBar {
    pub fn new(len: u64) -> crate::Result<Self> {
        let pb = indicatif::ProgressBar::new(len);
        pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
        ?
        .progress_chars("#>-"));
        Ok(Self(pb))
    }
}

impl Deref for ProgressBar {
    type Target = indicatif::ProgressBar;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ProgressBar {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
