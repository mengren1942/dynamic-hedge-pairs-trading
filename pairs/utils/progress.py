# pairs/utils/progress.py
from __future__ import annotations
from contextlib import contextmanager
import joblib  # only needed for the callback shim

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Redirect joblib progress to a given tqdm instance.

    Usage:
        from tqdm import tqdm
        from joblib import Parallel, delayed
        from pairs.utils.progress import tqdm_joblib

        with tqdm_joblib(tqdm(total=N)) as pbar:
            Parallel(n_jobs=...)(delayed(fn)(i) for i in range(N))
    """
    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()
