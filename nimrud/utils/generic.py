"""
useful things
"""


import numpy as np

def batcher(collection, chunk_size):
    """
    yield chunks of an incoming iterable. sequence or iterator is fine.
    """
    if isinstance(collection, np.ndarray) or isinstance(collection, list):
        for start in range(0, len(collection), chunk_size):
            stop = start + chunk_size
            yield collection[start: stop]
    else:
        iterator = iter(collection)
        try:
            while iterator:
                this_chunk = []
                for _ in range(chunk_size):
                    this_chunk.append(iterator.__next__())
                yield this_chunk
        except StopIteration:
            if this_chunk:
                yield this_chunk