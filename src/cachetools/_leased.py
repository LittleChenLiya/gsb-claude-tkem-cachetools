"""Lease layer for cachetools caches.

This module provides :class:`LeasedCache`, a wrapper around an
existing cache that adds explicit leases for cache entries, and
:class:`Lease`, a lease on a specific generation of a cache entry.
"""

import collections.abc
import threading

__all__ = ("Lease", "LeasedCache")


class Lease:
    """A lease on a cache entry of a specific generation.

    A lease is obtained by calling :meth:`LeasedCache.lease`.  It
    keeps the value of the leased entry alive even after the entry
    has been deleted from the cache or has expired, and defers
    resource cleanup until the lease is released.

    The lease can be used as a context manager; :meth:`release` is
    called automatically when the ``with`` block exits::

        with cache.lease(key) as lease:
            value = lease.value

    For convenience, a lease is iterable and unpacks to the
    ``(value, lease)`` pair, so the above can also be written as::

        with cache.lease(key) as (value, lease):
            ...

    """

    __slots__ = ("_cache", "_generation", "_key", "_released", "_value")

    def __init__(self, cache, key, generation, value):
        self._cache = cache
        self._key = key
        self._generation = generation
        self._value = value
        self._released = False

    @property
    def cache(self):
        """The cache this lease belongs to."""
        return self._cache

    @property
    def key(self):
        """The key of the leased cache entry."""
        return self._key

    @property
    def value(self):
        """The value of the leased cache entry."""
        return self._value

    @property
    def generation(self):
        """The generation of the leased cache entry."""
        return self._generation

    @property
    def released(self):
        """Whether this lease has already been released."""
        return self._released

    def release(self):
        """Release this lease.

        Returns :const:`True` if the lease was released by this call,
        :const:`False` if it had already been released.  Resource
        cleanup for the leased entry is triggered at most once, when
        the last lease of a generation is released.
        """
        return self._cache._release_lease(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def __iter__(self):
        yield self._value
        yield self

    def __repr__(self):
        return "<%s key=%r generation=%d%s>" % (
            type(self).__name__,
            self._key,
            self._generation,
            ", released" if self._released else "",
        )


class _Generation:
    """Internal state for a single generation of a cache entry."""

    __slots__ = ("alive", "cleaned", "leases", "value")

    def __init__(self, value):
        self.value = value
        self.leases = set()
        self.alive = True
        self.cleaned = False


class LeasedCache(collections.abc.MutableMapping):
    """Lease layer on top of a cachetools cache.

    Wraps an existing cache instance and adds explicit leases for
    cache entries.  A lease keeps the entry's value alive even after
    the entry has been deleted from the cache or has expired, and
    defers resource cleanup until the last lease of an entry's
    generation is released.

    Updating a cache entry via :meth:`__setitem__` creates a new
    generation.  Leases acquired for an older generation never affect
    the new generation: releasing an old lease only cleans up the old
    generation's value, never the new one.

    Cleanup callbacks are invoked outside the cache's internal lock,
    so the callback may safely re-enter the cache, e.g. to read cached
    values or acquire new leases, without risk of deadlock.

    Plain values without leases still follow the underlying cache's
    eviction rules; the lease layer does not interfere.

    :param cache: the underlying cache instance, e.g. an
        :class:`~cachetools.LRUCache` or :class:`~cachetools.TTLCache`.
    :param cleanup: an optional callable ``cleanup(key, value)`` that
        is invoked when a generation's resource is cleaned up.

    """

    __marker = object()

    def __init__(self, cache, cleanup=None):
        self.__cache = cache
        self.__cleanup = cleanup
        self.__lock = threading.RLock()
        self.__generations = {}
        self.__current = {}

    @property
    def cache(self):
        """The underlying cache instance."""
        return self.__cache

    def __getitem__(self, key):
        return self.__cache[key]

    def __setitem__(self, key, value):
        with self.__lock:
            was_present = key in self.__cache
            self.__cache[key] = value
            due = self.__retire(key, was_present)
            gens = self.__generations.setdefault(key, {})
            generation = (max(gens) + 1) if gens else 1
            gens[generation] = _Generation(value)
            self.__current[key] = generation
        self.__cleanup_due(due)

    def __delitem__(self, key):
        with self.__lock:
            if key not in self.__cache:
                raise KeyError(key)
            generation = self.__current.get(key)
            due = []
            if generation is not None:
                gen = self.__generations[key][generation]
                gen.alive = False
                self.__current[key] = None
                if not gen.leases:
                    del self.__generations[key][generation]
                    if not gen.cleaned:
                        gen.cleaned = True
                        due.append((key, gen.value))
            del self.__cache[key]
        self.__cleanup_due(due)

    def __contains__(self, key):
        return key in self.__cache

    def __iter__(self):
        return iter(self.__cache)

    def __len__(self):
        return len(self.__cache)

    def get(self, key, default=None):
        return self.__cache.get(key, default)

    def pop(self, key, default=__marker):
        with self.__lock:
            if key not in self.__cache:
                if default is self.__marker:
                    raise KeyError(key)
                return default
            value = self.__cache[key]
            generation = self.__current.get(key)
            due = []
            if generation is not None:
                gen = self.__generations[key][generation]
                gen.alive = False
                self.__current[key] = None
                if not gen.leases:
                    del self.__generations[key][generation]
                    if not gen.cleaned:
                        gen.cleaned = True
                        due.append((key, gen.value))
            del self.__cache[key]
        self.__cleanup_due(due)
        return value

    def popitem(self):
        with self.__lock:
            key, value = self.__cache.popitem()
            generation = self.__current.get(key)
            due = []
            if generation is not None:
                gen = self.__generations[key][generation]
                gen.alive = False
                self.__current[key] = None
                if not gen.leases:
                    del self.__generations[key][generation]
                    if not gen.cleaned:
                        gen.cleaned = True
                        due.append((key, gen.value))
        self.__cleanup_due(due)
        return key, value

    def clear(self):
        with self.__lock:
            due = []
            for key, generation in list(self.__current.items()):
                if generation is None:
                    continue
                gen = self.__generations[key][generation]
                gen.alive = False
                self.__current[key] = None
                if not gen.leases:
                    del self.__generations[key][generation]
                    if not gen.cleaned:
                        gen.cleaned = True
                        due.append((key, gen.value))
            self.__cache.clear()
        self.__cleanup_due(due)

    def lease(self, key):
        """Acquire a lease on the entry for `key`.

        Returns a :class:`Lease` for the current generation of the
        entry.  Raises :class:`KeyError` if the entry has been deleted
        from the cache, has expired, or has been evicted; no new
        leases are granted for such entries.

        """
        with self.__lock:
            generation = self.__current.get(key)
            if generation is None or key not in self.__cache:
                # entry deleted, expired or evicted: no new leases
                if generation is not None:
                    gen = self.__generations[key][generation]
                    gen.alive = False
                    self.__current[key] = None
                    if not gen.leases:
                        del self.__generations[key][generation]
                        # plain value evicted/expired: no cleanup
                raise KeyError(key)
            gen = self.__generations[key][generation]
            lease = Lease(self, key, generation, gen.value)
            gen.leases.add(lease)
            return lease

    def _release_lease(self, lease):
        with self.__lock:
            key = lease.key
            generation = lease.generation
            gens = self.__generations.get(key)
            if gens is None or generation not in gens:
                return False
            gen = gens[generation]
            if lease not in gen.leases:
                return False
            gen.leases.remove(lease)
            lease._released = True
            due = []
            if not gen.leases:
                # The generation is cleaned up only once it has been
                # removed from the underlying cache (deleted, evicted,
                # expired or replaced) and no leases remain.  If the
                # entry is still present in the cache, its resource is
                # still in use and cleanup is deferred.
                alive = (
                    gen.alive
                    and self.__current.get(key) == generation
                    and key in self.__cache
                )
                if not alive:
                    if not gen.cleaned:
                        gen.cleaned = True
                        due.append((key, gen.value))
                    del gens[generation]
                    if self.__current.get(key) == generation:
                        self.__current[key] = None
                    if not gens and self.__current.get(key) is None:
                        del self.__generations[key]
                        del self.__current[key]
        self.__cleanup_due(due)
        return True

    def __retire(self, key, was_present):
        # Retire the current generation of `key` in preparation for
        # updating the entry.  `was_present` indicates whether the key
        # was present in the underlying cache before the update.
        # Returns a list of (key, value) pairs due for immediate
        # cleanup.  A generation with outstanding leases is kept alive
        # in the lease layer and cleaned up later when its last lease
        # is released.  A generation without leases is only cleaned up
        # if the value was still present in the underlying cache,
        # i.e. it is being replaced; if it has already been evicted or
        # expired, it is discarded silently.  Cleanup happens at most
        # once per generation.
        generation = self.__current.get(key)
        if generation is None:
            return []
        gen = self.__generations[key][generation]
        gen.alive = False
        self.__current[key] = None
        if gen.leases:
            return []
        del self.__generations[key][generation]
        if gen.cleaned or not was_present:
            return []
        gen.cleaned = True
        return [(key, gen.value)]

    def __cleanup_due(self, due):
        # Invoke cleanup callbacks outside the internal lock, so the
        # callback may safely re-enter the cache without deadlocking.
        cleanup = self.__cleanup
        if cleanup is None:
            return
        for key, value in due:
            cleanup(key, value)
