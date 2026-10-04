"""Explicit lease layer on top of the cache classes.

This module provides :class:`LeaseCache`, a thread-safe wrapper around
any of the :mod:`cachetools` cache implementations, which hands out
explicit :class:`Lease` objects on reads.  A leased value is
guaranteed to remain usable until the lease is released, even if the
underlying cache item expires or is deleted in the meantime.  An
optional cleanup callback is invoked exactly once per value
*generation*, only after the value has left the cache and the last
lease on it has been released.
"""

import collections.abc
import threading

_MARKER = object()


class Lease:
    """A lease on a cached value.

    A lease pins one *generation* of a cached value: the value remains
    accessible via :attr:`value` until :meth:`release` is called, even
    if the corresponding cache item has expired or has been deleted.
    Leases are context managers, so they can be used with the
    :keyword:`with` statement.  Releasing a lease more than once is
    allowed and has no effect.

    """

    __slots__ = ("_gen", "_key", "_owner", "_released", "_value")

    def __init__(self, owner, key, gen, value):
        self._owner = owner
        self._key = key
        self._gen = gen
        self._value = value
        self._released = False

    def __repr__(self):
        state = "released" if self._released else "active"
        return "%s(key=%r, generation=%r, %s)" % (
            type(self).__name__,
            self._key,
            self._gen,
            state,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    @property
    def key(self):
        """The key this lease was acquired for."""
        return self._key

    @property
    def generation(self):
        """The generation of the cached value this lease refers to."""
        return self._gen

    @property
    def value(self):
        """The leased value.  Valid until the lease is released."""
        return self._value

    @property
    def released(self):
        """True if this lease has been released."""
        return self._released

    def release(self):
        """Release the lease.

        If this was the last active lease on a value generation that
        has already left the cache, the cleanup callback of the owning
        :class:`LeaseCache` is invoked for that value.  The callback
        is executed outside the internal lock, so it may safely
        re-access the cache.  Calling this method more than once is a
        no-op.

        """
        owner = self._owner
        with owner._lock:
            if self._released:
                return
            self._released = True
            pending = owner._release_locked(self._key, self._gen)
        owner._run_cleanups(pending)


class _Generation:
    """Internal bookkeeping for one generation of a cached value."""

    __slots__ = ("alive", "cleaned", "gen", "key", "leases", "value")

    def __init__(self, key, gen, value):
        self.key = key
        self.gen = gen
        self.value = value
        self.leases = 0  # number of active leases
        self.alive = True  # still present in the underlying cache
        self.cleaned = False  # cleanup callback already scheduled


class LeaseCache(collections.abc.MutableMapping):
    """Explicit lease layer on top of a :class:`Cache` instance.

    This wrapper delegates storage and eviction to an underlying
    cachetools cache, and adds explicit leases on reads.  Values
    written without leases are subject to the underlying cache's
    eviction rules as usual.  The optional `cleanup` callback is
    invoked as ``cleanup(key, value)`` exactly once per value
    generation: either when the value leaves the cache while no leases
    are active, or when the last lease on an already removed value is
    released.  The callback always runs outside the internal lock and
    may re-access this cache.

    """

    def __init__(self, cache, cleanup=None):
        self._cache = cache
        self._cleanup = cleanup
        self._lock = threading.Lock()
        self._gens = {}  # key -> current generation number
        self._entries = {}  # (key, gen) -> _Generation

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self._cache)

    @property
    def cache(self):
        """The underlying cache."""
        return self._cache

    @property
    def cleanup(self):
        """The cleanup callback, or None."""
        return self._cleanup

    def lease(self, key):
        """Acquire a lease on the value currently stored for `key`.

        Raises :exc:`KeyError` if `key` is not cached or its item has
        expired.  Once an item has expired or been deleted, no new
        leases on it are handed out; existing leases remain usable
        until released.

        """
        with self._lock:
            pending = []
            entry = self._sync_locked(key, pending)
            if entry is not None:
                entry.leases += 1
                lease = Lease(self, key, entry.gen, entry.value)
            else:
                lease = None
        self._run_cleanups(pending)
        if lease is None:
            raise KeyError(key)
        return lease

    def get_lease(self, key):
        """Acquire a lease and return a ``(value, lease)`` pair.

        Raises :exc:`KeyError` if `key` is not cached or its item has
        expired.

        """
        lease = self.lease(key)
        return lease.value, lease

    def __getitem__(self, key):
        with self._lock:
            pending = []
            entry = self._sync_locked(key, pending)
        self._run_cleanups(pending)
        if entry is None:
            raise KeyError(key)
        return entry.value

    def __setitem__(self, key, value):
        with self._lock:
            pending = []
            entry = self._sync_locked(key, pending)
            if entry is not None:
                # replacing the current generation
                self._retire_locked(entry, pending)
                gen = entry.gen + 1
            else:
                gen = self._gens.get(key, -1) + 1
            self._gens[key] = gen
            self._entries[(key, gen)] = _Generation(key, gen, value)
            # may evict other items from the underlying cache
            self._cache[key] = value
            self._reap_locked(pending)
        self._run_cleanups(pending)

    def __delitem__(self, key):
        with self._lock:
            pending = []
            entry = self._sync_locked(key, pending)
            if entry is not None:
                del self._cache[key]
                self._retire_locked(entry, pending)
        self._run_cleanups(pending)
        if entry is None:
            raise KeyError(key)

    def __contains__(self, key):
        return key in self._cache

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def popitem(self):
        """Remove and return a `(key, value)` pair from the underlying cache."""
        with self._lock:
            pending = []
            key, value = self._cache.popitem()
            gen = self._gens.get(key)
            entry = self._entries.get((key, gen)) if gen is not None else None
            if entry is not None and entry.alive and entry.value is value:
                self._retire_locked(entry, pending)
        self._run_cleanups(pending)
        return (key, value)

    def clear(self):
        with self._lock:
            pending = []
            self._cache.clear()
            for entry in list(self._entries.values()):
                if entry.alive:
                    self._retire_locked(entry, pending)
        self._run_cleanups(pending)

    def expire(self, time=None):
        """Expire items in the underlying cache, if it supports it.

        Returns the iterable of expired `(key, value)` pairs reported
        by the underlying cache, or an empty list if the underlying
        cache does not provide an :meth:`expire` method.

        """
        expire = getattr(self._cache, "expire", None)
        if expire is None:
            return []
        with self._lock:
            pending = []
            expired = expire(time)
            self._reap_locked(pending)
        self._run_cleanups(pending)
        return expired

    @property
    def maxsize(self):
        """The maximum size of the underlying cache."""
        return self._cache.maxsize

    @property
    def currsize(self):
        """The current size of the underlying cache."""
        return self._cache.currsize

    # internal helpers; methods ending in ``_locked`` must be called
    # with ``self._lock`` held and never invoke the cleanup callback

    def _sync_locked(self, key, pending):
        """Reconcile the tracked state of `key` with the underlying cache.

        Returns the live :class:`_Generation` entry for `key`, or None
        if `key` is not currently cached.

        """
        gen = self._gens.get(key, -1)
        entry = self._entries.get((key, gen))
        value = self._cache.get(key, _MARKER)
        if entry is not None and entry.alive:
            if value is entry.value:
                return entry
            # expired, evicted, or replaced underneath us
            self._retire_locked(entry, pending)
        if value is _MARKER:
            return None
        # adopt an untracked value stored directly in the cache
        gen += 1
        self._gens[key] = gen
        entry = _Generation(key, gen, value)
        self._entries[(key, gen)] = entry
        return entry

    def _reap_locked(self, pending):
        """Retire all tracked generations no longer present in the cache."""
        cache_get = self._cache.get
        for entry in list(self._entries.values()):
            if entry.alive and cache_get(entry.key, _MARKER) is not entry.value:
                self._retire_locked(entry, pending)

    def _retire_locked(self, entry, pending):
        """Mark a generation as removed from the underlying cache."""
        entry.alive = False
        if entry.leases == 0 and not entry.cleaned:
            entry.cleaned = True
            del self._entries[(entry.key, entry.gen)]
            if self._cleanup is not None:
                pending.append(entry)

    def _release_locked(self, key, gen):
        """Decrement the lease count of a generation; called by Lease."""
        entry = self._entries.get((key, gen))
        if entry is None:
            return []
        entry.leases -= 1
        pending = []
        if entry.leases == 0 and not entry.alive and not entry.cleaned:
            entry.cleaned = True
            del self._entries[(key, gen)]
            if self._cleanup is not None:
                pending.append(entry)
        return pending

    def _run_cleanups(self, pending):
        """Invoke the cleanup callback; always called without the lock."""
        cleanup = self._cleanup
        if cleanup is None:
            return
        for entry in pending:
            cleanup(entry.key, entry.value)
