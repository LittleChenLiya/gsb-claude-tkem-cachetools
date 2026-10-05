"""Tests for TTLCache.expire_batch.

Run with a virtual (manually advanced) clock::

    python -m pytest tests/test_ttl_expire_batch.py -v
"""

import unittest

from cachetools import TTLCache


class VirtualClock:
    """A manually advanced clock for deterministic TTL tests.

    The clock only advances when ``advance`` is called, so TTL expiry
    is fully under the test's control.
    """

    def __init__(self):
        self.time = 0

    def __call__(self):
        return self.time

    def advance(self, seconds=1):
        self.time += seconds


class TTLExpireBatchTest(unittest.TestCase):
    def test_expire_batch_basic(self):
        """Five expired items, capacity 2, in three batches."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        for i in range(5):
            cache[i] = i
        # clock stays at 0; the scan uses an explicit expiry time

        expired, cursor = cache.expire_batch(batchsize=2, time=2)
        self.assertEqual([(0, 0), (1, 1)], expired)
        self.assertIsNotNone(cursor)

        # update one item during the scan; it must be retained
        cache[3] = 3
        self.assertIn(3, cache)

        expired, cursor = cache.expire_batch(batchsize=2, cursor=cursor)
        self.assertEqual([(2, 2), (4, 4)], expired)
        self.assertIsNotNone(cursor)

        expired, cursor = cache.expire_batch(batchsize=2, cursor=cursor)
        self.assertEqual([], expired)
        self.assertIsNone(cursor)

        # the updated item is retained; everything else is gone
        self.assertEqual({3}, set(cache))
        self.assertEqual(1, len(cache))

    def test_expire_batch_updated_item_retained(self):
        """An item updated during the scan is not removed by the old cursor."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        for i in range(4):
            cache[i] = i

        expired, cursor = cache.expire_batch(batchsize=2, time=2)
        self.assertEqual(2, len(expired))

        # update a non-cursor item during the scan; it must be retained
        cache[3] = 3
        self.assertIn(3, cache)

        expired, cursor = cache.expire_batch(batchsize=2, cursor=cursor)
        # item 2 is still expired; item 3 was updated (new generation) and
        # is retained by the old cursor
        self.assertEqual([(2, 2)], expired)
        self.assertIsNone(cursor)
        self.assertIn(3, cache)

    def test_expire_batch_new_item_not_deleted(self):
        """An item added during the scan is not removed by the old cursor."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        for i in range(3):
            cache[i] = i

        expired, cursor = cache.expire_batch(batchsize=2, time=2)
        self.assertEqual(2, len(expired))

        # add a new item during the scan
        cache[9] = 9
        self.assertIn(9, cache)

        expired, cursor = cache.expire_batch(batchsize=2, cursor=cursor)
        # item 2 is still expired; the new item (9) has a newer generation
        # and is retained by the old cursor
        self.assertEqual([(2, 2)], expired)
        self.assertIsNone(cursor)
        self.assertIn(9, cache)

    def test_expire_batch_deleted_item_skipped(self):
        """Items deleted during the scan are skipped."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        for i in range(4):
            cache[i] = i

        expired, cursor = cache.expire_batch(batchsize=2, time=2)
        self.assertEqual(2, len(expired))

        # delete the item the cursor is about to visit
        del cache[2]
        self.assertNotIn(2, cache)

        expired, cursor = cache.expire_batch(batchsize=2, cursor=cursor)
        # item 2 was deleted; item 3 is still expired
        self.assertEqual([(3, 3)], expired)
        self.assertIsNone(cursor)

    def test_expire_batch_fixed_scan_time(self):
        """The scan time is fixed when the scan starts."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        cache[1] = 1
        cache[2] = 2

        # start the scan at time 2
        expired, cursor = cache.expire_batch(batchsize=1, time=2)
        self.assertEqual([(1, 1)], expired)

        # advance the clock past item 2's expiry; the scan still uses time=2
        clock.advance(1)
        expired, cursor = cache.expire_batch(batchsize=1, cursor=cursor)
        # item 2 expires at 2, scan time is fixed at 2 -> still expired
        self.assertEqual([(2, 2)], expired)
        self.assertIsNone(cursor)

    def test_expire_batch_cursor_binds_cache(self):
        """A cursor from one cache cannot be used on another cache."""
        clock1 = VirtualClock()
        cache1 = TTLCache(maxsize=10, ttl=2, timer=clock1)
        cache1[1] = 1
        cache1[2] = 2
        _, cursor = cache1.expire_batch(batchsize=1, time=2)
        self.assertIsNotNone(cursor)

        clock2 = VirtualClock()
        cache2 = TTLCache(maxsize=10, ttl=2, timer=clock2)
        with self.assertRaises(ValueError):
            cache2.expire_batch(batchsize=1, cursor=cursor)

    def test_expire_batch_consumed_cursor_rejected(self):
        """Reusing a consumed cursor is explicitly rejected."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        cache[1] = 1
        cache[2] = 2
        _, cursor = cache.expire_batch(batchsize=1, time=2)
        self.assertIsNotNone(cursor)

        # first use consumes the cursor
        cache.expire_batch(batchsize=1, cursor=cursor)
        # second use (reuse) is rejected
        with self.assertRaises(ValueError):
            cache.expire_batch(batchsize=1, cursor=cursor)

    def test_expire_batch_callback_reentrant_read(self):
        """The callback runs outside the lock and may reentrantly read."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        for i in range(3):
            cache[i] = i

        read_during_callback = []

        def callback(key, value):
            # reentrant read: access the cache while the callback runs
            read_during_callback.append((key, cache.get(key), len(cache)))

        expired, _cursor = cache.expire_batch(batchsize=2, time=2, callback=callback)
        self.assertEqual(2, len(expired))
        # the callback was invoked for each expired item without deadlock
        self.assertEqual(2, len(read_during_callback))
        for key, _value, size in read_during_callback:
            self.assertIsNone(cache.get(key))
            self.assertEqual(1, size)

    def test_expire_batch_callback_outside_lock(self):
        """The callback is invoked outside the cache's internal lock."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        for i in range(2):
            cache[i] = i

        reentrant = []

        def callback(key, value):
            # if the callback were called under the cache's lock, this
            # reentrant call would deadlock; here it runs outside the lock
            expired, cursor = cache.expire_batch(batchsize=1, time=2)
            reentrant.append((expired, cursor))

        expired, _cursor = cache.expire_batch(batchsize=2, time=2, callback=callback)
        self.assertEqual(2, len(expired))
        # the reentrant calls completed without deadlock
        self.assertEqual(2, len(reentrant))

    def test_expire_batch_invalid_batchsize(self):
        """batchsize must be a positive integer."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        with self.assertRaises(ValueError):
            cache.expire_batch(batchsize=0)
        with self.assertRaises(ValueError):
            cache.expire_batch(batchsize=-1)

    def test_expire_batch_empty_cache(self):
        """Scanning an empty cache returns no items and no cursor."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        expired, cursor = cache.expire_batch(batchsize=2, time=2)
        self.assertEqual([], expired)
        self.assertIsNone(cursor)

    def test_expire_batch_no_expired_items(self):
        """Scanning with no expired items returns nothing."""
        clock = VirtualClock()
        cache = TTLCache(maxsize=10, ttl=2, timer=clock)
        cache[1] = 1
        expired, cursor = cache.expire_batch(batchsize=2, time=0)
        self.assertEqual([], expired)
        self.assertIsNone(cursor)
        self.assertIn(1, cache)


if __name__ == "__main__":
    unittest.main()
