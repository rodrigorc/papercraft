//! An semaphore with associated data.
//!
//! I'm sure there are many crates out there that do something similar,
//! but it is easy enough to do it myself, custom designed for this project.
//!
//! The initial count of the semaphore will be the number of cpus times a factor,
//! 1 by default.

use std::sync::{Condvar, Mutex};

pub struct Semaphore<D> {
    inner: Mutex<Data<D>>,
    cvar: Condvar,
}

struct Data<D> {
    count: usize,
    data: D,
}

impl<D: Default> Default for Semaphore<D> {
    fn default() -> Self {
        Self::new_with_cpu_num(D::default())
    }
}

impl<D> Semaphore<D> {
    pub fn new_with_cpu_num(data: D) -> Self {
        Self::new_with_n_times_cpu_num(1, data)
    }

    pub fn new_with_n_times_cpu_num(n: usize, data: D) -> Self {
        let count = n * num_cpus::get();
        Self::new(count, data)
    }

    pub fn new(count: usize, data: D) -> Self {
        Semaphore {
            inner: Mutex::new(Data { count, data }),
            cvar: Condvar::new(),
        }
    }

    pub fn wait(&self, count: usize) {
        self.wait_with(count, |_| ());
    }

    pub fn wait_with<R>(&self, count: usize, f: impl FnOnce(&mut D) -> R) -> R {
        let inner = self.inner.lock().unwrap();
        let mut inner = self.cvar.wait_while(inner, |d| d.count < count).unwrap();
        inner.count -= count;
        f(&mut inner.data)
    }

    pub fn signal(&self, count: usize) {
        self.signal_with(count, |_| ());
    }

    pub fn signal_with<R>(&self, count: usize, f: impl FnOnce(&mut D) -> R) -> R {
        let mut inner = self.inner.lock().unwrap();
        inner.count += count;
        let res = f(&mut inner.data);
        self.cvar.notify_one();
        res
    }

    pub fn into_inner(self) -> D {
        self.inner.into_inner().unwrap().data
    }
}
