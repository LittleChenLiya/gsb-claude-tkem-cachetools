"""Partitioned cache adapter: map keys to independent partitions.

Each partition has its own capacity and eviction order, so a hot
partition cannot evict another partition's items.
"""

import collections

from . import Cache, LRUCache


__all__ = ("PartitionedCache", "Partition")


class Partition(collections.namedtuple("Partition", ["maxsize", "cachecls"])):
    """Partition configuration: ``(maxsize, cachecls)``.

    ``maxsize`` is the partition's capacity; ``cachecls`` is the cache
    class used for eviction (default: :class:`LRUCache`).
    """

    __slots__ = ()

    def __new__(cls, maxsize, cachecls=LRUCache):
        if maxsize < 0:
            raise ValueError("maxsize must be non-negative")
        return super().__new__(cls, maxsize, cachecls)


class _PartitionState:
    """Internal mutable state for a single partition."""

    __slots__ = ("name", "maxsize", "cachecls", "cache")

    def __init__(self, name, maxsize, cachecls, getsizeof):
        self.name = name
        self.maxsize = maxsize
        self.cachecls = cachecls
        self.cache = cachecls(maxsize, getsizeof)

    def update(self, maxsize=None, cachecls=None):
        """Update the partition configuration, evicting to fit.

        Recreates the sub-cache with the new configuration and transfers
        items in eviction order.  The new cache evicts as needed, so the
        result matches lowering the old cache's capacity in place.
        """
        if maxsize is not None:
            if maxsize < 0:
                raise ValueError("maxsize must be non-negative")
            self.maxsize = maxsize
        if cachecls is not None:
            self.cachecls = cachecls
        old = self.cache
        new = self.cachecls(self.maxsize, old.getsizeof)
        while old:
            key, value = old.popitem()
            try:
                new[key] = value
            except ValueError:
                # Item too large for the new capacity: it is evicted,
                # leaving no partial state in the new cache.
                pass
        self.cache = new


class PartitionedCache(Cache):
    """Cache adapter that maps keys to independent partitions.

    Each partition has its own capacity and eviction order, so a hot
    partition cannot evict another partition's items.

    Args:
        partitions: Mapping of partition name to configuration.  The
            configuration is either an integer ``maxsize``, a
            ``(maxsize, cachecls)`` tuple, or a :class:`Partition`
            instance.
        partitionfunc: Callable mapping a key to a partition name.
        getsizeof: Callable returning the size of a value (default: 1).

    Example:
        >>> from cachetools import PartitionedCache, LRUCache, FIFOCache
        >>> cache = PartitionedCache(
        ...     partitions={"A": 10, "B": (20, FIFOCache)},
        ...     partitionfunc=lambda key: "A" if key < 100 else "B",
        ... )
        >>> cache[1] = "a"
        >>> cache[200] = "b"
    """

    __marker = object()

    def __init__(self, partitions, partitionfunc, getsizeof=None):
        Cache.__init__(self, 0, getsizeof)
        self.__partitionfunc = partitionfunc
        self.__states = collections.OrderedDict()
        for name, config in partitions.items():
            maxsize, cachecls = self._parse_config(config)
            self.__states[name] = _PartitionState(
                name, maxsize, cachecls, self.getsizeof
            )

    @staticmethod
    def _parse_config(config):
        if isinstance(config, Partition):
            return config.maxsize, config.cachecls
        if isinstance(config, collections.abc.Mapping):
            return config["maxsize"], config.get("cachecls", LRUCache)
        if isinstance(config, (tuple, list)):
            if len(config) >= 2:
                return config[0], config[1]
            return config[0], LRUCache
        return config, LRUCache

    def _state(self, key):
        """Return the partition state for ``key``, calling partitionfunc once."""
        name = self.__partitionfunc(key)
        try:
            return self.__states[name]
        except KeyError:
            raise KeyError("unknown partition %r" % (name,)) from None

    def __getitem__(self, key):
        return self._state(key).cache[key]

    def __setitem__(self, key, value):
        self._state(key).cache[key] = value

    def __delitem__(self, key):
        del self._state(key).cache[key]

    def __contains__(self, key):
        return key in self._state(key).cache

    def __iter__(self):
        for state in self.__states.values():
            yield from state.cache

    def __len__(self):
        return sum(len(state.cache) for state in self.__states.values())

    @property
    def currsize(self):
        return sum(state.cache.currsize for state in self.__states.values())

    @property
    def maxsize(self):
        return sum(state.maxsize for state in self.__states.values())

    def get(self, key, default=None):
        return self._state(key).cache.get(key, default)

    def pop(self, key, default=__marker):
        state = self._state(key)
        if default is self.__marker:
            return state.cache.pop(key)
        return state.cache.pop(key, default)

    def setdefault(self, key, default=None):
        return self._state(key).cache.setdefault(key, default)

    def popitem(self):
        """Remove and return a ``(key, value)`` pair from the first non-empty
        partition (in partition name order)."""
        for state in self.__states.values():
            if state.cache:
                return state.cache.popitem()
        raise KeyError("%s is empty" % type(self).__name__)

    def clear(self):
        for state in self.__states.values():
            state.cache.clear()

    def __repr__(self):
        parts = ", ".join(
            "%s(%s/%s)" % (name, state.cache.currsize, state.maxsize)
            for name, state in self.__states.items()
        )
        return "%s(%s)" % (type(self).__name__, parts)

    def get_partition(self, name):
        """Return the sub-cache for partition ``name``."""
        return self.__states[name].cache

    def set_partition(self, name, maxsize=None, cachecls=None):
        """Update a partition's configuration, evicting to fit.

        Generates a new configuration and evicts items according to the
        partition's own eviction order.
        """
        self.__states[name].update(maxsize, cachecls)

    def partition_names(self):
        """Return the list of partition names."""
        return list(self.__states)
