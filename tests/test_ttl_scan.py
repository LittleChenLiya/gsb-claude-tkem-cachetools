"""Tests for ``TTLCache.expire_scan()`` batched expiration.

All tests use a virtual clock so they are fully deterministic.
"""

import pickle

import pytest

from cachetools import TTLCache


class VirtualClock:
    """A deterministic, manually advanced clock."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds=1.0):
        self.now += seconds


@pytest.fixture
def clock():
    return VirtualClock()


@pytest.fixture
def cache(clock):
    return TTLCache(maxsize=100, ttl=10, timer=clock)


def scan_all(cache, count, cursor=None, callback=None):
    """Drain a scan, returning all expired (key, value) pairs."""
    expired = []
    while True:
        batch, cursor = cache.expire_scan(count, cursor, callback=callback)
        expired.extend(batch)
        if cursor is None:
            return expired


def test_five_expired_items_expire_in_three_batches_of_two(cache, clock):
    for i in range(5):
        cache[i] = i
    clock.advance(10)  # all five items are now expired

    batch1, cursor = cache.expire_scan(2)
    assert batch1 == [(0, 0), (1, 1)]
    assert cursor is not None

    batch2, cursor = cache.expire_scan(2, cursor)
    assert batch2 == [(2, 2), (3, 3)]
    assert cursor is not None

    batch3, cursor = cache.expire_scan(2, cursor)
    assert batch3 == [(4, 4)]
    assert cursor is None  # scan complete

    assert len(cache) == 0


def test_scan_of_empty_cache_completes_immediately(cache):
    expired, cursor = cache.expire_scan(2)
    assert expired == []
    assert cursor is None


def test_scan_ignores_unexpired_items(cache, clock):
    cache["old"] = 1
    clock.advance(5)
    cache["young"] = 2
    clock.advance(5)  # t=10: "old" expired, "young" expires at t=15

    expired, cursor = cache.expire_scan(10)
    assert expired == [("old", 1)]
    assert cursor is None
    assert cache["young"] == 2


def test_scan_time_is_fixed_at_scan_start(cache, clock):
    cache["a"] = 1  # expires at t=10
    cache["b"] = 2  # expires at t=10
    clock.advance(5)
    cache["c"] = 3  # expires at t=15
    clock.advance(5)  # t=10: scan starts now

    batch, cursor = cache.expire_scan(1)
    assert batch == [("a", 1)]
    assert cursor is not None
    assert cursor.time == 10

    clock.advance(10)  # t=20: "c" has expired in real time by now...

    # ...but the scan's reference time is still fixed at t=10
    expired = scan_all(cache, 1, cursor)
    assert expired == [("b", 2)]
    # "c" was not expired at the scan's fixed reference time
    assert set(cache._TTLCache__links) == {"c"}


def test_item_updated_during_scan_is_kept(cache, clock):
    for i in range(5):
        cache[i] = i  # inserted at t=0, expire at t=10
    clock.advance(5)  # t=5: nothing has expired in real time yet

    # scan with a fixed reference time of t=10, at which all five
    # items will have expired
    batch, cursor = cache.expire_scan(2, time=10)
    assert batch == [(0, 0), (1, 1)]

    # update an item the scan has not reached yet, and insert a new
    # one; both belong to a newer generation than the scan's horizon
    cache[2] = "updated"
    cache["new"] = "new"

    batch, cursor = cache.expire_scan(2, cursor)
    assert batch == [(3, 3), (4, 4)]
    batch, cursor = cache.expire_scan(2, cursor)
    assert batch == []
    assert cursor is None

    # the updated and the new item were kept by the old cursor
    assert cache[2] == "updated"
    assert cache["new"] == "new"
    assert len(cache) == 2


def test_newer_generation_is_never_removed_by_old_cursor(cache, clock):
    cache["a"] = 1
    cache["b"] = 2
    clock.advance(10)

    _, cursor = cache.expire_scan(1)  # removes "a"
    cache["b"] = 3  # "b" now belongs to a newer generation

    # force the updated item to look expired at the scan's fixed time;
    # the generation guard must still protect it from the old cursor
    cache._TTLCache__links["b"].expires = 0

    expired, cursor = cache.expire_scan(1, cursor)
    assert expired == []
    assert cursor is None
    assert set(cache._TTLCache__links) == {"b"}


def test_item_deleted_during_scan_is_skipped(cache, clock):
    for i in range(4):
        cache[i] = i
    clock.advance(5)
    cache["keep"] = "keep"  # expires at t=15
    clock.advance(5)  # t=10: items 0..3 are expired, "keep" is not

    batch, cursor = cache.expire_scan(1)
    assert batch == [(0, 0)]

    # items vanish by other means before the scan reaches them
    del cache["keep"]  # unexpired items can still be deleted
    removed = cache.expire()  # normal expire() removes 1, 2, 3 at once
    assert len(removed) == 3

    # the in-progress scan simply skips whatever is already gone
    expired, cursor = cache.expire_scan(2, cursor)
    assert expired == []
    assert cursor is None
    assert len(cache) == 0


def test_consumed_cursor_is_rejected(cache, clock):
    for i in range(3):
        cache[i] = i
    clock.advance(10)

    _, cursor = cache.expire_scan(1)
    cache.expire_scan(1, cursor)  # consumes the cursor
    with pytest.raises(ValueError, match="already been consumed"):
        cache.expire_scan(1, cursor)


def test_cursor_is_bound_to_cache_identity(cache, clock):
    other = TTLCache(maxsize=100, ttl=10, timer=clock)
    for i in range(3):
        cache[i] = i
        other[i] = i
    clock.advance(10)

    _, cursor = cache.expire_scan(1)
    with pytest.raises(ValueError, match="different cache"):
        other.expire_scan(1, cursor)
    # the original cache still accepts its own cursor
    batch, _ = cache.expire_scan(1, cursor)
    assert batch == [(1, 1)]


def test_invalid_cursor_is_rejected(cache):
    with pytest.raises(ValueError, match="different cache"):
        cache.expire_scan(1, object())


def test_count_must_be_positive(cache):
    with pytest.raises(ValueError, match="positive"):
        cache.expire_scan(0)


def test_callback_is_called_for_each_expired_item(cache, clock):
    for i in range(3):
        cache[i] = i
    clock.advance(10)

    calls = []
    expired, cursor = cache.expire_scan(3, callback=lambda k, v: calls.append((k, v)))
    assert calls == [(0, 0), (1, 1), (2, 2)]
    assert calls == expired
    assert cursor is None


def test_callback_runs_outside_the_scan_lock(cache, clock):
    for i in range(3):
        cache[i] = i
    clock.advance(10)

    def callback(key, value):
        # the scan lock must not be held while callbacks run
        assert cache._TTLCache__scan_lock.acquire(blocking=False)
        cache._TTLCache__scan_lock.release()
        # re-entrant reads (and even nested scans) must not deadlock
        assert cache.get(key) is None
        assert key not in cache
        cache.expire_scan(1)

    expired, _ = cache.expire_scan(3, callback=callback)
    assert expired == [(0, 0), (1, 1), (2, 2)]


def test_normal_expire_is_unchanged(cache, clock):
    for i in range(5):
        cache[i] = i
    clock.advance(10)
    expired = cache.expire()  # still removes everything in one pass
    assert expired == [(i, i) for i in range(5)]
    assert len(cache) == 0


def test_pickle_roundtrip_with_scan_state(cache, clock):
    for i in range(3):
        cache[i] = i
    clock.advance(10)
    _, cursor = cache.expire_scan(1)

    # unpickling expires stale items, but must recreate the scan lock
    clone = pickle.loads(pickle.dumps(cache))
    assert clone == cache
    # a cursor is bound to the identity of its originating cache
    with pytest.raises(ValueError, match="different cache"):
        clone.expire_scan(1, cursor)
    # ...but the clone can run its own scans
    expired, cursor = clone.expire_scan(2)
    assert expired == []
    assert cursor is None
