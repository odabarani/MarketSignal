"""Bounded caches that expire while a long-running app stays open."""

from functools import lru_cache, wraps
import time


def ttl_cache(seconds=900, maxsize=16):
    def decorate(function):
        @lru_cache(maxsize=maxsize)
        def cached(bucket, *args, **kwargs):
            return function(*args, **kwargs)

        @wraps(function)
        def wrapped(*args, **kwargs):
            return cached(int(time.monotonic() // seconds), *args, **kwargs)

        wrapped.cache_clear = cached.cache_clear
        wrapped.cache_info = cached.cache_info
        return wrapped
    return decorate
