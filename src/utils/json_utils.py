import numpy as np

def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    # tuples are JSON-serializable as lists if needed; default will leave tuples as tuples,
    # but to be explicit you can uncomment the next two lines:
    # if isinstance(obj, tuple):
    #     return list(obj)
    return obj