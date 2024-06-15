use self_update::cargo_crate_version;

pub fn update() -> crate::Result<()> {
    use self_update::update::UpdateStatus;
    let status = self_update::backends::github::Update::configure()
        .repo_owner("0x676e67")
        .repo_name("fcsrv")
        .bin_name("fcsrv")
        .target(self_update::get_target())
        .show_output(true)
        .show_download_progress(true)
        .no_confirm(true)
        .current_version(cargo_crate_version!())
        .build()?
        .update_extended()?;
    if let UpdateStatus::Updated(ref release) = status {
        if let Some(body) = &release.body {
            if !body.trim().is_empty() {
                println!("fcsrv upgraded to {}:\n", release.version);
                println!("{}", body);
            } else {
                println!("fcsrv upgraded to {}", release.version);
            }
        }
    } else {
        println!("fcsrv is up-to-date");
    }

    Ok(())
}
