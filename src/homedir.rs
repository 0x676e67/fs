#[cfg(not(target_os = "windows"))]
static USER_HOME: OnceLock<PathBuf> = OnceLock::new();

#[cfg(not(target_os = "windows"))]
pub fn setting_dir(dir: PathBuf) {
    let _ = USER_HOME.get_or_init(|| dir);
}

#[cfg(target_os = "windows")]
mod home_dir_windows {
    use core::slice::from_raw_parts;
    use std::{
        ffi::{c_void, OsString},
        os::windows::prelude::OsStringExt,
        path::PathBuf,
        ptr::null_mut,
    };
    use windows_sys::Win32::{
        Globalization::lstrlenW,
        System::Com::CoTaskMemFree,
        UI::Shell::{FOLDERID_Profile, SHGetKnownFolderPath},
    };

    /// Return the user's home directory.
    ///
    /// ```
    /// //  "C:\Users\USER"
    /// let path = simple_home_dir::home_dir().unwrap();
    /// ```
    pub fn home_dir() -> Option<PathBuf> {
        let mut path_ptr = null_mut();
        (unsafe { SHGetKnownFolderPath(&FOLDERID_Profile, 0, 0, &mut path_ptr) } == 0).then_some({
            let wide = unsafe { from_raw_parts(path_ptr, lstrlenW(path_ptr) as usize) };
            let ostr = OsString::from_wide(wide);
            unsafe { CoTaskMemFree(path_ptr as *const c_void) }
            ostr.into()
        })
    }
}

#[cfg(not(target_os = "windows"))]
mod home_dir_ne_windows {
    use std::{env::var_os, path::PathBuf};

    const HOME: &str = "HOME";

    /// Return the user's home directory.
    ///
    /// ```
    /// //  "/home/USER"
    /// let path = simple_home_dir::home_dir().unwrap();
    /// ```
    pub fn home_dir() -> Option<PathBuf> {
        if let Some(home) = super::USER_HOME.get() {
            return Some(home.clone());
        }
        var_os(HOME).map(Into::into)
    }
}

#[cfg(not(target_os = "windows"))]
pub use home_dir_ne_windows::*;
#[cfg(target_os = "windows")]
pub use home_dir_windows::*;
use std::{path::PathBuf, sync::OnceLock};
