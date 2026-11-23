"""Host-only publication timing. No tensor inspection or device synchronization."""
import contextvars
import json
import os
import threading
import time
from contextlib import contextmanager

_active = contextvars.ContextVar('publication_host', default=None)


@contextmanager
def span(label):
    record = _active.get()
    if record is None:
        yield
        return
    begin = time.monotonic_ns()
    status = 'ok'
    try:
        yield
    except BaseException:
        status = 'exception'
        raise
    finally:
        end = time.monotonic_ns()
        # Materialization may occur hundreds of times; aggregate without names/data.
        if label == 'lazy_materialize':
            item = record['materialize']
            item['count'] += 1
            item['host_ns'] += end-begin
            item['exceptions'] += status != 'ok'
            item['first_begin_ns'] = item['first_begin_ns'] or begin
            item['last_end_ns'] = end
        else:
            record['spans'].append({'phase':label,'begin_ns':begin,'end_ns':end,'status':status})


@contextmanager
def publication(step, rank, mode):
    directory = os.environ.get('OPD_PUBLICATION_HOST_DIR')
    if not directory:
        yield
        return
    record = {'schema':1,'scope':'host API intervals; nested spans not additive',
              'step':step,'rank':rank,'mode':mode,'pid':os.getpid(),
              'tid':threading.get_ident(),'begin_ns':time.monotonic_ns(),
              'spans':[],'status':'ok','materialize':{'count':0,'host_ns':0,
              'exceptions':0,'first_begin_ns':None,'last_end_ns':None}}
    token = _active.set(record)
    try:
        yield
    except BaseException:
        record['status'] = 'exception'
        raise
    finally:
        record['end_ns'] = time.monotonic_ns()
        _active.reset(token)
        # Fail open for telemetry I/O: never replace an application exception/result.
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f'publication-{record["pid"]}-{record["tid"]}.jsonl')
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, separators=(',', ':'))+'\n')
        except Exception:
            pass
