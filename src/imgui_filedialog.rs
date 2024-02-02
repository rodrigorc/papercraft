#![allow(dead_code)]

use bitflags::bitflags;
use easy_imgui_window::easy_imgui as imgui;
use std::ffi::CString;

mod fd {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(clippy::all)]

    // Opaque pointer, bindgen cannot build it
    pub type ImGuiFileDialog = std::ffi::c_void;
    use easy_imgui_sys::*;
    include!(concat!(env!("OUT_DIR"), "/imgui_filedialog_bindings.rs"));
}

pub struct FileDialog {
    ptr: *mut fd::ImGuiFileDialog,
}

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
        const SHOW_READ_ONLY_CHECK = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_ShowReadOnlyCheck;
        const DEFAULT = fd::ImGuiFileDialogFlags__ImGuiFileDialogFlags_Default;
    }
    pub struct ResultMode: fd::IGFD_ResultMode_ {
        const AddIfNoFileExt = fd::IGFD_ResultMode__IGFD_ResultMode_AddIfNoFileExt;
        const OverwriteFileExt = fd::IGFD_ResultMode__IGFD_ResultMode_OverwriteFileExt;
        const KeepInputFile = fd::IGFD_ResultMode__IGFD_ResultMode_KeepInputFile;
    }
}

pub struct Builder<'a> {
    key: &'a str,
    title: &'a str,
    filter: &'a str,
    path: &'a str,
    file: &'a str,
    count_selection_max: i32,
    flags: Flags,
}
impl<'a> Builder<'a> {
    #[must_use]
    pub fn new(key: &'a str) -> Builder<'a> {
        Builder {
            key,
            title: "",
            filter: "",
            path: "",
            file: "",
            count_selection_max: 1,
            flags: Flags::DEFAULT,
        }
    }
    pub fn title(mut self, title: &'a str) -> Self {
        self.title = title;
        self
    }
    pub fn filter(mut self, filter: &'a str) -> Self {
        self.filter = filter;
        self
    }
    pub fn path(mut self, path: &'a str) -> Self {
        self.path = path;
        self
    }
    pub fn file(mut self, file: &'a str) -> Self {
        self.file = file;
        self
    }
    pub fn flags(mut self, flags: Flags) -> Self {
        self.flags = flags;
        self
    }
    pub fn open(self) -> FileDialog {
        let key = CString::new(self.key).unwrap();
        let title = CString::new(self.title).unwrap();
        let filter = CString::new(self.filter).unwrap();
        let path = CString::new(self.path).unwrap();
        let file = CString::new(self.file).unwrap();
        let ptr;
        unsafe {
            ptr = fd::IGFD_Create();
            fd::IGFD_OpenDialog(
                ptr,
                key.as_ptr(),
                title.as_ptr(),
                filter.as_ptr(),
                path.as_ptr(),
                file.as_ptr(),
                self.count_selection_max,
                std::ptr::null_mut(),
                self.flags.bits() as _,
            );
        };
        FileDialog { ptr }
    }
}

impl FileDialog {
    pub fn is_opened(&self) -> bool {
        unsafe { fd::IGFD_IsOpened(self.ptr) }
    }
    pub fn display<'a>(
        &'a mut self,
        key: &str,
        flags: imgui::WindowFlags,
        min_size: imgui::Vector2,
        max_size: imgui::Vector2,
    ) -> Option<DisplayToken<'a>> {
        let key = CString::new(key).unwrap();
        let ok = unsafe {
            fd::IGFD_DisplayDialog(
                self.ptr,
                key.as_ptr(),
                flags.bits() as _,
                imgui::v2_to_im(min_size),
                imgui::v2_to_im(max_size),
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
        unsafe {
            fd::IGFD_Destroy(self.ptr);
        }
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
        std::str::from_utf8(cstr.to_bytes())
            .ok()
            .map(String::from)
            .unwrap_or_default()
    }
}

impl DisplayToken<'_> {
    pub fn ok(&self) -> bool {
        unsafe { fd::IGFD_IsOk(self.fd.ptr) }
    }
    pub fn readonly(&self) -> bool {
        unsafe { fd::IGFD_IsReadonly(self.fd.ptr) }
    }
    pub fn file_path_name(&self, mode: ResultMode) -> Option<String> {
        unsafe {
            let sz = fd::IGFD_GetFilePathName(self.fd.ptr, mode.bits() as i32);
            opt_str_from_ptr(sz)
        }
    }
    pub fn current_file_name(&self, mode: ResultMode) -> Option<String> {
        unsafe {
            let sz = fd::IGFD_GetCurrentFileName(self.fd.ptr, mode.bits() as i32);
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
    pub fn selection(&self, mode: ResultMode) -> Selection {
        let sel = unsafe { fd::IGFD_GetSelection(self.fd.ptr, mode.bits() as i32) };
        Selection { sel }
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
        SelectionIterator { sel: self, pos: 0 }
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
