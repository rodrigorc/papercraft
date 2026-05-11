//! An semaphore with associated data.
//!
//! I'm sure there are many crates out there that do something similar,
//! but it is easy enough to do it myself, custom designed for this project.
//!
//! The initial count of the semaphore will be the number of cpus times a factor,
//! 1 by default.

use std::sync::{Condvar, Mutex};

pub struct Semaphore<T> {
    inner: Mutex<Data<T>>,
    cvar: Condvar,
}

struct Data<T> {
    count: usize,
    data: T,
}

impl<T: Default> Default for Semaphore<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> Semaphore<T> {
    pub fn new(t: T) -> Self {
        Self::new_with_n_times(1, t)
    }

    pub fn new_with_n_times(n: usize, t: T) -> Self {
        Semaphore {
            inner: Mutex::new(Data {
                count: n * num_cpus::get(),
                data: t,
            }),
            cvar: Condvar::new(),
        }
    }

    pub fn take(&self, count: usize) {
        self.take_with(count, |_| ());
    }

    pub fn take_with<R>(&self, count: usize, f: impl FnOnce(&mut T) -> R) -> R {
        let inner = self.inner.lock().unwrap();
        let mut inner = self.cvar.wait_while(inner, |d| d.count < count).unwrap();
        inner.count -= count;
        log::debug!("sema down to {}", inner.count);
        f(&mut inner.data)
    }

    pub fn release(&self, count: usize) {
        self.release_with(count, |_| ());
    }

    pub fn release_with<R>(&self, count: usize, f: impl FnOnce(&mut T) -> R) -> R {
        let mut inner = self.inner.lock().unwrap();
        inner.count += count;
        log::debug!("sema up to {}", inner.count);
        let res = f(&mut inner.data);
        self.cvar.notify_one();
        res
    }

    pub fn into_inner(self) -> T {
        self.inner.into_inner().unwrap().data
    }
}
