//! An semaphore with associated data.
//!
//! I'm sure there are many crates out there that do something similar,
//! but it is easy enough to do it myself, custom designed for this project.
//!
//! The initial count of the semaphore will be the number of cpus times a factor,
//! 1 by default.
#![allow(dead_code)]

use std::{
    sync::{Condvar, Mutex},
    time::Duration,
};

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

    pub fn wait_timeout(&self, count: usize, timeout: Duration) -> Option<()> {
        self.wait_timeout_with(count, timeout, |_| ())
    }

    pub fn wait_timeout_with<R>(
        &self,
        count: usize,
        timeout: Duration,
        f: impl FnOnce(&mut D) -> R,
    ) -> Option<R> {
        let inner = self.inner.lock().unwrap();
        let (mut inner, to) = self
            .cvar
            .wait_timeout_while(inner, timeout, |d| d.count < count)
            .unwrap();
        if to.timed_out() {
            None
        } else {
            inner.count -= count;
            Some(f(&mut inner.data))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    #[test]
    fn test_count() {
        let sem = Semaphore::new(2, ());
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), Some(()));
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), Some(()));
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), None);

        sem.signal(1);
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), Some(()));
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), None);

        sem.signal(2);
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), Some(()));
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), Some(()));
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), None);

        sem.signal(2);
        assert_matches!(sem.wait_timeout(3, Duration::ZERO), None);

        assert_matches!(sem.wait_timeout(2, Duration::ZERO), Some(()));
        assert_matches!(sem.wait_timeout(1, Duration::ZERO), None);
    }

    #[test]
    fn test_limits() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let sem = Semaphore::new(4, ());
        let active_threads = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for _ in 0..20 {
                s.spawn(|| {
                    for _ in 0..10 {
                        sem.wait(1);

                        let current = active_threads.fetch_add(1, Ordering::SeqCst) + 1;
                        assert!(current <= 4, "Too many threads: {}", current);

                        // Simulate work
                        std::thread::sleep(Duration::from_millis(1));

                        active_threads.fetch_sub(1, Ordering::SeqCst);
                        sem.signal(1);

                        std::thread::sleep(Duration::from_millis(1));
                    }
                });
            }
        });
    }

    #[test]
    fn test_timeout() {
        let sem = Semaphore::new(0, ());
        let result = sem.wait_timeout(1, Duration::from_millis(10));
        assert!(result.is_none());
    }

    #[test]
    fn test_timeout_closure() {
        let sem = Semaphore::new(3, 0);
        let result = sem.wait_timeout_with(1, Duration::from_millis(10), |data| *data += 1);
        assert!(result.is_some());
        assert_eq!(sem.into_inner(), 1);
    }

    #[test]
    fn test_timeout_closure_fail() {
        let sem = Semaphore::new(0, 0);
        let result = sem.wait_timeout_with(1, Duration::from_millis(10), |data| *data += 1);
        assert!(result.is_none());
        assert_eq!(sem.into_inner(), 0);
    }

    #[test]
    fn test_wait_closure() {
        let sem = Semaphore::new(1, 0);
        let result = sem.wait_with(1, |data| {
            *data += 1;
            *data
        });
        assert_eq!(result, 1);
        assert_eq!(sem.into_inner(), 1);
    }

    #[test]
    fn test_signal_notifies() {
        let sem = Semaphore::new(0, ());
        let sem_ref = &sem;
        std::thread::scope(|s| {
            s.spawn(move || {
                std::thread::sleep(Duration::from_millis(10));
                sem_ref.signal(1);
            });
            let result = sem.wait_timeout(1, Duration::from_millis(100));
            assert!(result.is_some());
        });
    }
}
