//! Tables of canonical values.

use std::borrow::{Borrow, ToOwned};
use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::RwLock;

/// A synchronized set of canonical values of one type.
///
/// Every canonical value lives as long as the `SyncInterner` that contains it.
/// Values are never removed individually.
#[derive(Debug)]
pub struct SyncInterner<T>
where
    T: Hash + Eq + ToOwned + ?Sized,
{
    // Note: RwLock currently (Q3 2026) uses an atomic reader count on platforms
    // with futexes, and the high rate of updates to the reader count can induce
    // cache contention as core count increases. The SyncInterner API can
    // theoretically be implemented without it, using an atomic hash tree or
    // similar approach. For now, we'll just stick to the standard library and
    // use the poorly-scaling RwLock as a proof of concept for the API.
    table: RwLock<HashSet<OwnedBox<T>>>,
}

impl<T> SyncInterner<T>
where
    T: Hash + Eq + ToOwned + ?Sized,
{
    /// Creates an empty interner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the canonical instance of `value`, copying it if necessary.
    pub fn intern(&self, value: &T) -> &T {
        if let Some(existing) = self.get(value) {
            return existing;
        }

        // Another thread may insert the same value between get's read lock and
        // this write lock, so the second lookup is required.
        let mut table = self.table.write().unwrap();
        let pointer = match table.get(value) {
            Some(existing) => existing.borrow() as *const T,
            None => {
                let owned = OwnedBox::new(value.to_owned());
                let pointer = owned.borrow() as *const T;
                let was_absent = table.insert(owned);
                debug_assert!(was_absent);
                pointer
            }
        };

        // We never drop entries from the table,
        // so we know that they will live as long as self.
        unsafe { &*pointer }
    }

    /// Returns the canonical instance of `value`, if it has been interned.
    pub fn get(&self, value: &T) -> Option<&T> {
        let table = self.table.read().unwrap();
        let pointer = table.get(value)?.borrow() as *const T;

        // We never drop entries from the table,
        // so we know that they will live as long as self.
        Some(unsafe { &*pointer })
    }
}

impl<T> Default for SyncInterner<T>
where
    T: Hash + Eq + ToOwned + ?Sized,
{
    fn default() -> Self {
        Self {
            table: RwLock::new(HashSet::new()),
        }
    }
}

/// A box containing `T::Owned` that hashes and compares as `T`.
struct OwnedBox<T>
where
    T: ToOwned + ?Sized,
{
    value: Box<T::Owned>,
}

impl<T> OwnedBox<T>
where
    T: ToOwned + ?Sized,
{
    fn new(value: T::Owned) -> Self {
        Self {
            value: Box::new(value),
        }
    }
}

impl<T> Borrow<T> for OwnedBox<T>
where
    T: ToOwned + ?Sized,
{
    fn borrow(&self) -> &T {
        self.value.deref().borrow()
    }
}

impl<T> Debug for OwnedBox<T>
where
    T: ToOwned + Debug + ?Sized,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value: &T = self.borrow();
        value.fmt(formatter)
    }
}

impl<T> Hash for OwnedBox<T>
where
    T: ToOwned + Hash + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let value: &T = self.borrow();
        value.hash(state);
    }
}

impl<T> PartialEq for OwnedBox<T>
where
    T: ToOwned + PartialEq + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        let value: &T = self.borrow();
        value.eq(other.borrow())
    }
}

impl<T> Eq for OwnedBox<T> where T: ToOwned + Eq + ?Sized {}

#[cfg(test)]
mod tests {
    use super::SyncInterner;
    use std::thread;

    #[test]
    fn equal_values_have_the_same_address() {
        let interner = SyncInterner::<str>::new();
        let x = &"foobarbar"[3..6];
        let y = &"barbeque"[..3];

        assert_eq!(x, y);
        assert_ne!(x.as_ptr(), y.as_ptr());

        let interned_x = interner.intern(x);
        let interned_y = interner.intern(y);
        assert_eq!(interned_x.as_ptr(), interned_y.as_ptr());
        assert_ne!(interned_x.as_ptr(), x.as_ptr());
    }

    #[test]
    fn interned_value_outlives_input() {
        let interner = SyncInterner::new();
        let interned = {
            let input = String::from("temporary");
            interner.intern(input.as_str())
        };

        assert_eq!(interned, "temporary");
        assert_eq!(interner.get("temporary"), Some(interned));
    }

    #[test]
    fn equal_values_have_the_same_address_across_threads() {
        let interner = SyncInterner::<str>::new();
        let (x, y) = thread::scope(|scope| {
            let x = scope.spawn(|| interner.intern("shared").as_ptr() as usize);
            let y = scope.spawn(|| interner.intern("shared").as_ptr() as usize);
            (x.join().unwrap(), y.join().unwrap())
        });

        assert_eq!(x, y);
    }

    #[test]
    fn missing_value_is_not_found() {
        let interner = SyncInterner::<str>::new();
        assert_eq!(interner.get("missing"), None);
    }
}
