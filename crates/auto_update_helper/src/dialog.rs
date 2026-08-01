//! A minimal Win32 progress window shown while `auto_update_helper` swaps
//! files. Same general shape as Zed's dialog (a plain WNDCLASS with a
//! progress bar control), written independently for StealCode.

use anyhow::{Context, Result};
use windows::{
    Win32::{
        Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM},
        Graphics::Gdi::{
            BeginPaint, EndPaint, PAINTSTRUCT, ReleaseDC, TextOutW,
        },
        System::LibraryLoader::GetModuleHandleW,
        UI::{
            Controls::{PBM_SETRANGE, PBM_SETSTEP, PBM_STEPIT, PROGRESS_CLASS},
            WindowsAndMessaging::{
                CreateWindowExW, DefWindowProcW, GetDesktopWindow,
                GetWindowRect, PostMessageW, PostQuitMessage, RegisterClassW,
                SendMessageW, WINDOW_EX_STYLE, WM_CLOSE, WM_CREATE, WM_DESTROY,
                WM_PAINT, WM_USER, WNDCLASSW, WS_CAPTION, WS_CHILD,
                WS_EX_TOPMOST, WS_POPUP, WS_VISIBLE,
            },
        },
    },
    core::HSTRING,
};

pub(crate) const WM_JOB_UPDATED: u32 = WM_USER + 1;
pub(crate) const WM_TERMINATE: u32 = WM_USER + 2;

static mut PROGRESS_BAR: HWND = HWND(std::ptr::null_mut());

pub fn create_dialog_window(total_steps: usize) -> Result<HWND> {
    unsafe {
        let class_name = windows::core::w!("StealCode-Update-Dialog");
        let module =
            GetModuleHandleW(None).context("unable to get module handle")?;
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            lpszClassName: class_name,
            hInstance: module.into(),
            ..Default::default()
        };
        RegisterClassW(&wc);

        let mut rect = RECT::default();
        GetWindowRect(GetDesktopWindow(), &mut rect)
            .context("unable to get desktop rect")?;
        let (width, height) = (400, 120);

        let hwnd = CreateWindowExW(
            WS_EX_TOPMOST,
            class_name,
            windows::core::w!("StealCode"),
            WS_VISIBLE | WS_POPUP | WS_CAPTION,
            rect.right / 2 - width / 2,
            rect.bottom / 2 - height / 2,
            width,
            height,
            None,
            None,
            None,
            None,
        )
        .context("unable to create dialog window")?;

        let progress_bar = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            PROGRESS_CLASS,
            None,
            WS_CHILD | WS_VISIBLE,
            20,
            50,
            340,
            30,
            Some(hwnd),
            None,
            None,
            None,
        )
        .context("unable to create progress bar")?;
        SendMessageW(
            progress_bar,
            PBM_SETRANGE,
            None,
            Some(LPARAM((total_steps.max(1) * 10) as isize)),
        );
        SendMessageW(progress_bar, PBM_SETSTEP, Some(WPARAM(10)), None);
        PROGRESS_BAR = progress_bar;

        Ok(hwnd)
    }
}

pub fn notify_job_done(hwnd: HWND) {
    unsafe {
        let _ = PostMessageW(Some(hwnd), WM_JOB_UPDATED, WPARAM(0), LPARAM(0));
    }
}

pub fn notify_terminate(hwnd: isize) {
    unsafe {
        let _ = PostMessageW(
            Some(HWND(hwnd as _)),
            WM_TERMINATE,
            WPARAM(0),
            LPARAM(0),
        );
    }
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_PAINT => unsafe {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);
            let _ =
                TextOutW(hdc, 20, 15, &HSTRING::from("Updating StealCode..."));
            let _ = EndPaint(hwnd, &ps);
            ReleaseDC(Some(hwnd), hdc);
            LRESULT(0)
        },
        m if m == WM_JOB_UPDATED => unsafe {
            SendMessageW(PROGRESS_BAR, PBM_STEPIT, None, None);
            LRESULT(0)
        },
        m if m == WM_TERMINATE => unsafe {
            PostQuitMessage(0);
            LRESULT(0)
        },
        WM_CLOSE => LRESULT(0),
        WM_DESTROY => unsafe {
            PostQuitMessage(0);
            LRESULT(0)
        },
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}
