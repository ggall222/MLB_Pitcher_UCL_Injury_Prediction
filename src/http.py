"""Immutable response bodies, atomic cache writes, bounded retry and provenance."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import logging
import time
import uuid
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)

class FetchError(RuntimeError):
    pass

class HttpClient:
    def __init__(self, cfg, offline=False, refresh=False):
        self.cfg, self.offline, self.refresh = cfg, offline, refresh
        self.root = Path(cfg['cache_path'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = []

    def get(self, url, params=None, mutable=False, force_refresh=False):
        prepared = requests.Request('GET', url, params=params).prepare().url
        key = hashlib.sha256(prepared.encode()).hexdigest()
        index = self.root / (key + '.json')
        meta = json.loads(index.read_text()) if index.exists() else None
        fresh = (meta is not None and (not mutable or
                 time.time() - meta['cached_at'] < self.cfg['refresh_metadata_hours'] * 3600))
        if meta and (self.offline or (fresh and not self.refresh and not force_refresh)):
            body_path = self.root / meta['body_file']
            if body_path.exists():
                body = gzip.decompress(body_path.read_bytes())
                if hashlib.sha256(body).hexdigest() != meta['sha256']:
                    raise FetchError('Cache checksum mismatch: ' + str(body_path))
                self.records.append(dict(meta, cache_hit=True))
                return body
        if self.offline:
            raise FetchError('Offline cache miss: ' + prepared)
        session = requests.Session()
        retry = Retry(total=self.cfg['request_retries'], backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET'])
        session.mount('https://', HTTPAdapter(max_retries=retry))
        try:
            response = session.get(prepared, timeout=(15, self.cfg['request_timeout_seconds']),
                                   headers={'User-Agent': 'mlb-ucl-capstone/0.1 (academic research)'})
            response.raise_for_status()
            body = response.content
        except requests.RequestException as exc:
            self.records.append({'url': prepared, 'error': str(exc), 'cache_hit': False})
            raise FetchError(str(exc)) from exc
        finally:
            session.close()
        sha = hashlib.sha256(body).hexdigest()
        body_path = self.root / (sha + '.gz')
        if not body_path.exists():
            tmp = self.root / (uuid.uuid4().hex + '.tmp')
            tmp.write_bytes(gzip.compress(body))
            tmp.replace(body_path)
        meta = {'url': prepared, 'sha256': sha, 'body_file': body_path.name,
                'cached_at': time.time(), 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'bytes': len(body), 'status': response.status_code}
        tmp = self.root / (uuid.uuid4().hex + '.tmp')
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(index)
        self.records.append(dict(meta, cache_hit=False))
        return body

    def json(self, url, params=None, mutable=False):
        try:
            return json.loads(self.get(url, params, mutable))
        except (ValueError, UnicodeError) as exc:
            raise FetchError('Invalid JSON: ' + url) from exc

    def many(self, jobs, fn):
        """Bound in-flight responses: completed seasons cannot accumulate in RAM."""
        iterator = iter(jobs)
        with ThreadPoolExecutor(max_workers=self.cfg['network_workers']) as pool:
            futures = {}
            def replenish():
                while len(futures) < self.cfg['network_workers'] * 2:
                    try:
                        job = next(iterator)
                    except StopIteration:
                        break
                    futures[pool.submit(fn, job)] = job
            replenish()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        LOG.warning('Source item %s failed: %s', job, exc)
                        yield job, None, str(exc)
                    else:
                        yield job, result, None
                replenish()
