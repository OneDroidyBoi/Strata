"""Opt-in request reservations using explicit, operator-calibrated peak costs."""
from serve.resource_admission import AdmissionRejected, ResourceLedger


class RequestAdmission:
    FIELDS = frozenset({
        'host_capacity_bytes', 'gpu_capacity_bytes', 'gpu_id', 'max_requests',
        'fixed_host_bytes', 'fixed_gpu_bytes', 'host_request_bytes', 'gpu_request_bytes',
        'host_bytes_per_token', 'gpu_bytes_per_token', 'max_context_tokens', 'lookahead_tokens'})

    def __init__(self, config):
        if not isinstance(config, dict) or set(config) != self.FIELDS:
            raise ValueError('resource_admission requires exactly: ' + ', '.join(sorted(self.FIELDS)))
        for key, value in config.items():
            if key == 'gpu_id':
                if not isinstance(value, str) or not value:
                    raise ValueError('gpu_id must be a nonempty string')
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(key + ' must be a nonnegative integer')
        if not config['max_context_tokens'] or not config['max_requests']:
            raise ValueError('context and request limits must be positive')
        self.config = dict(config)
        self.ledger = ResourceLedger(config['host_capacity_bytes'],
                                     {config['gpu_id']: config['gpu_capacity_bytes']})
        self.worker = self.ledger.register_worker(
            'lane', config['gpu_id'], config['max_requests'],
            config['fixed_host_bytes'], config['fixed_gpu_bytes'])

    def acquire(self, prompt_tokens, max_new):
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            raise ValueError('prompt_tokens must be a nonnegative integer')
        if isinstance(max_new, bool) or not isinstance(max_new, int):
            raise ValueError('max_new must be an integer')
        c = self.config
        tokens = prompt_tokens + max_new + c['lookahead_tokens'] if max_new > 0 else c['max_context_tokens']
        if prompt_tokens + c['lookahead_tokens'] > c['max_context_tokens'] or tokens > c['max_context_tokens']:
            raise AdmissionRejected('request exceeds configured context budget')
        return self.ledger.acquire(
            self.worker, host_bytes=c['host_request_bytes'] + tokens * c['host_bytes_per_token'],
            gpu_bytes=c['gpu_request_bytes'] + tokens * c['gpu_bytes_per_token'], context_tokens=tokens)

    def run(self, factory, prompt_tokens, max_new, quiescent):
        lease = self.acquire(prompt_tokens, max_new)
        iterator = None
        close_ok = False
        failed = False
        try:
            iterator = factory()
            # yield-from implicitly closes its delegate; a second close can
            # hide a failure of the first native cancellation drain.
            for event in iterator:
                yield event
        except GeneratorExit:
            raise
        except BaseException:
            failed = True
            raise
        finally:
            try:
                if iterator is not None:
                    iterator.close()
                close_ok = True
            finally:
                try:
                    drained = close_ok and not failed and quiescent() is True
                except Exception:
                    drained = False
                if drained:
                    self.ledger.release(lease, iterator_closed=True, work_drained=True)
                else:
                    # Never reuse uncertain capacity after an incomplete cancellation drain.
                    self.gate_admission()

    def gate_admission(self):
        self.ledger.begin_drain(self.worker)

    def wait_drained(self, timeout_s=0):
        return self.ledger.wait_quiescent(self.worker, timeout=timeout_s)


def configured_admission(cfg, gpu_ids):
    if 'resource_admission' not in cfg:
        return None
    if cfg.get('replicas') not in (None, False, 1) or len(gpu_ids) != 1:
        raise ValueError('resource_admission currently requires one single-GPU lane')
    if cfg.get('vision') or cfg.get('images'):
        raise ValueError('resource_admission does not yet account for sidecar models')
    controller = RequestAdmission(cfg['resource_admission'])
    if controller.config['gpu_id'] != str(gpu_ids[0]):
        raise ValueError('resource_admission gpu_id must match the configured GPU')
    return controller
