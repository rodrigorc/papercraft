#![allow(dead_code)]

use std::ffi::CString;
use bitflags::bitflags;

mod fd {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/imgui_filedialog_bindings.rs"));

    impl From<super::Vector2> for ImVec2 {
        fn from(v: super::Vector2) -> ImVec2 {
            ImVec2 {
                x: v.x,
                y: v.y,
            }
        }
    }
}

pub struct FileDialog {
    ptr: *mut fd::ImGuiFileDialog,
}

type Vector2 = mint::Vector2<f32>;

bitflags! {
    pub struct Flags: fd::ImGuiFileDialogFlags_ {
        const NONE = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_None;
        const CONFIRM_OVERWRITE = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_ConfirmOverwrite;
        const DONT_SHOW_HIDDEN_FILES = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_DontShowHiddenFiles;
        const DISABLE_CREATE_DIRECTORY_BUTTON = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_DisableCreateDirectoryButton;
        const HIDE_COLUMN_TYPE = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_HideColumnType;
        const HIDE_COLUMN_SIZE = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_HideColumnSize;
        const HIDE_COLUMN_DATE = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_HideColumnDate;
        const NO_DIALOG = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_NoDialog;
        const READ_ONLY_FILE_NAME_FIELD = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_ReadOnlyFileNameField;
        const CASE_INSENSITIVE_EXTENSION = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_CaseInsensitiveExtention;
        const MODAL = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_Modal;
        const DEFAULT = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_Default;
    }
}

impl FileDialog {
    pub fn new() -> FileDialog {
        let ptr = unsafe {
            fd::IGFD_Create()
        };
        FileDialog {
            ptr,
        }
    }
    pub fn is_opened(&self) -> bool {
        unsafe {
            fd::IGFD_IsOpened(self.ptr)
        }
    }
    pub fn open(&mut self, key: &str, title: &str, filter: &str, path: &str, file: &str,
            count_selection_max: i32, flags: Flags) {
        let key = CString::new(key).unwrap();
        let title = CString::new(title).unwrap();
        let filter = CString::new(filter).unwrap();
        let path = CString::new(path).unwrap();
        let file = CString::new(file).unwrap();
        unsafe {
            fd::IGFD_OpenDialog(
                self.ptr,
                key.as_ptr(),
                title.as_ptr(),
                filter.as_ptr(),
                path.as_ptr(),
                file.as_ptr(),
                count_selection_max,
                std::ptr::null_mut(),
                flags.bits() as _
            )
        }
    }
    pub fn display<'a>(&'a mut self, key: &str, flags: imgui::WindowFlags, min_size: impl Into<Vector2>, max_size: impl Into<Vector2>) -> Option<DisplayToken<'a>> {
        let key = CString::new(key).unwrap();
        let ok = unsafe {
            fd::IGFD_DisplayDialog(
                self.ptr,
                key.as_ptr(),
                flags.bits() as _,
                min_size.into().into(),
                max_size.into().into(),
            )
        };
        if ok {
            Some(DisplayToken { fd: self })
        } else {
            None
        }
    }
}

impl Drop for FileDialog {
    fn drop(&mut self) {
        unsafe { fd::IGFD_Destroy(self.ptr); }
    }
}

pub struct DisplayToken<'a> {
    fd: &'a mut FileDialog,
}

unsafe fn opt_str_from_ptr(sz: *const std::ffi::c_char) -> Option<String> {
    if sz.is_null() {
        None
    } else {
        let cstr = std::ffi::CStr::from_ptr(sz);
        let res = std::str::from_utf8(cstr.to_bytes()).ok().map(String::from);
        fd::free(sz as _);
        res
    }
}
unsafe fn str_from_ptr(sz: *const std::ffi::c_char) -> String {
    if sz.is_null() {
        String::new()
    } else {
        let cstr = std::ffi::CStr::from_ptr(sz);
        std::str::from_utf8(cstr.to_bytes()).ok().map(String::from).unwrap_or_default()
    }
}

impl DisplayToken<'_> {
    pub fn ok(&self) -> bool {
        unsafe {
            fd::IGFD_IsOk(self.fd.ptr)
        }
    }
    pub fn file_path_name(&self) -> Option<String> {
        unsafe {
            let sz = fd::IGFD_GetFilePathName(self.fd.ptr);
            opt_str_from_ptr(sz)
        }
    }
    pub fn current_file_name(&self) -> Option<String> {
        unsafe {
            let sz = fd::IGFD_GetCurrentFileName(self.fd.ptr);
            opt_str_from_ptr(sz)
        }
    }
    pub fn current_path(&self) -> Option<String> {
        unsafe {
            let sz = fd::IGFD_GetCurrentPath(self.fd.ptr);
            opt_str_from_ptr(sz)
        }
    }
    pub fn current_filter(&self) -> Option<String> {
        unsafe {
            let sz = fd::IGFD_GetCurrentFilter(self.fd.ptr);
            opt_str_from_ptr(sz)
        }
    }
    pub fn selection(&self) -> Selection {
        let sel = unsafe {
            fd::IGFD_GetSelection(self.fd.ptr)
        };
        Selection {
            sel,
        }
    }
    pub fn close(self) {
        unsafe {
            fd::IGFD_CloseDialog(self.fd.ptr);
        }
    }
}

pub struct Selection {
    sel: fd::IGFD_Selection,
}

impl<'a> IntoIterator for &'a Selection {
    type IntoIter = SelectionIterator<'a>;
    type Item = (String, String);
    fn into_iter(self) -> Self::IntoIter {
        SelectionIterator {
            sel: self,
            pos: 0,
        }
    }
}

impl Drop for Selection {
    fn drop(&mut self) {
        unsafe {
            fd::IGFD_Selection_DestroyContent(&mut self.sel);
        }
    }
}

pub struct SelectionIterator<'a> {
    sel: &'a Selection,
    pos: usize,
}

impl Iterator for SelectionIterator<'_> {
    type Item = (String, String);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.sel.sel.count {
            None
        } else {
            unsafe {
                let entry = &*self.sel.sel.table.add(self.pos);
                let s1 = str_from_ptr(entry.fileName);
                let s2 = str_from_ptr(entry.filePathName);
                self.pos += 1;
                Some((s1, s2))
            }
        }
    }
}


