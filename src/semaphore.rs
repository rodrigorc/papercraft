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
    fn test_thread() {
        struct Tok;

        // Keep the sem count and the vec len the same.
        let sem = Semaphore::new(4, vec![Tok, Tok, Tok, Tok]);
        let chrono = std::time::Instant::now();
        std::thread::scope(|s| {
            for _ in 0..10 {
                s.spawn({
                    let sem = &sem;
                    move || {
                        for _ in 0..10 {
                            // If the semaphore fails to arbitrate, panic!
                            let tok = sem.wait_with(1, |v| v.pop()).expect("should have a Tok");
                            std::thread::sleep(Duration::from_millis(10));
                            sem.signal_with(1, |v| v.push(tok));

                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                });
            }
        });
        let elapsed = chrono.elapsed().as_millis();

        dbg!(elapsed);
        // 10 threads touching tokens 10 times each. Those are 100 touches.
        // Each touch is 10 ms. Total 1000 ms.
        // With 4 tokens, it should take a bit over 250 ms (3 tokens would be about 333 ms)
        // Allow for some imprecision and overloads.
        // I know, tests that measure time are not pretty, but it works for me...
        assert!(249 < elapsed && elapsed < 300);
    }
}
