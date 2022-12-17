#!/usr/bin/env python
from __future__ import annotations

import functools
import logging


def log_call(func: callable) -> callable:
    """ log the function name on call with level debug. """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"{func.__name__} was called")
        return func(*args, **kwargs)
    return wrapper
