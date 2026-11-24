"""Reserve conservative request costs before each HTTP attempt; no secrets saved."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4


class BudgetExceeded(RuntimeError):
    pass


class RequestBudgetExceeded(BudgetExceeded):
    """Only the current task exceeded the per-request input admission limit."""
    pass


class BoundedTransport:
    """Serial pilot ledger, including failed/retried HTTP attempts.

    Reserves UTF-8 payload bytes + 8192 as a conservative input token estimate,
    and output max_tokens at a conservative price. This is an admission bound,
    not an invoice or provider-tokenizer measurement. Reservations never fall.
    """
    def __init__(self, ledger, *, total_limit=50.0, run_limit=5.0,
                 input_price=0.8, output_price=8.0, max_input_estimate=64000,
                 sender=None, unit="yuan", event_log_path=None):
        if unit not in {"yuan", "credits", "tokens"}:
            raise ValueError("unknown budget unit")
        self.unit = unit
        self.reservation_field = "reserved_" + unit
        self.ledger = Path(ledger)
        self.total_limit, self.run_limit = total_limit, run_limit
        self.input_price, self.output_price = input_price, output_price
        self.max_input_estimate = max_input_estimate
        self.run_reserved = 0.0
        self.calls = []
        self.sender = sender or self._send
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self.current_task_id = None

    def _event(self, event, row):
        if self.event_log_path is None:
            return
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **row}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _send(url, payload, headers, timeout):
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def __call__(self, url, payload, headers, timeout):
        if payload.get("enable_thinking") is not False:
            raise ValueError("pilot requires explicit non-thinking requests")
        estimated_input = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 8192
        output = payload.get("max_tokens")
        if type(output) is not int or not 1 <= output <= 512:
            raise ValueError("pilot output limit must be between 1 and 512")
        if estimated_input > self.max_input_estimate:
            raise RequestBudgetExceeded("request admission input estimate exceeded")
        reservation = estimated_input + output if self.unit == "tokens" else (estimated_input * self.input_price + output * self.output_price) / 1_000_000
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        lock = self.ledger.with_suffix(".lock")
        # A crash can leave a lock: stop for review rather than risk overspend.
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            state = json.loads(self.ledger.read_text(encoding="utf-8")) if self.ledger.exists() else {self.reservation_field: 0.0, "http_attempts": 0, "unit": self.unit}
            if self.reservation_field not in state or state.get("unit", self.unit) != self.unit:
                raise ValueError("budget ledger unit mismatch; existing ledger unchanged")
            if state[self.reservation_field] + reservation > self.total_limit or self.run_reserved + reservation > self.run_limit:
                raise BudgetExceeded("currency reservation limit exceeded; no request sent")
            state[self.reservation_field] += reservation
            state["http_attempts"] += 1
            temporary = self.ledger.with_suffix(".pending")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.ledger)
        finally:
            os.close(descriptor)
            lock.unlink()
        self.run_reserved += reservation
        row = {"attempt_id": uuid4().hex, "task_id": self.current_task_id,
               "estimated_input_upper": estimated_input, "output_limit": output, self.reservation_field: reservation, "status": "pending"}
        self.calls.append(row)
        self._event("started", row)
        try:
            response = self.sender(url, payload, headers, timeout)
            row.update(status="success", model=response.get("model"), usage=response.get("usage"))
            return response
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, http_status=getattr(exc, "code", None))
            raise
        finally:
            self._event("finished", row)
