# edge/services/event_publisher.py
#
# MİMARİ KARAR: Outbox Pattern
#
# Network koptuğunda event kaybolmamalı.
# Outbox Pattern: event önce lokal kuyruğa yazılır,
# sonra merkeze iletilir. İletim başarısız olursa tekrar denenir.
# Bu "at-least-once delivery" garantisi verir.
#
# KAVRAM: Observer Pattern (bağlantı)
# Publisher event'leri yayınlar, subscriber'lar dinler.
# Publisher subscriber'ların kim olduğunu bilmez.

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Optional
import time
import json
import threading
import logging

from edge.models.detection_models import FrameSonucu, BagajTespiti

logger = logging.getLogger(__name__)


@dataclass
class OutboxEntry:
    """
    Outbox kuyruğundaki bir girdi.

    KAVRAM: Immutable ID + mutable state
    entry_id ve payload değişmez. retry_count ve status değişir.
    """
    entry_id:    str
    gate_id:     str
    payload:     dict
    created_at:  float = field(default_factory=time.time)
    retry_count: int   = 0
    status:      str   = "pending"   # pending | sent | failed

    MAX_RETRY = 3

    @property
    def is_expired(self) -> bool:
        """3 başarısız deneme → kalıcı hata."""
        return self.retry_count >= self.MAX_RETRY


class DeadLetterQueue:
    """
    Başarısız event'lerin son durağı.

    İletilemeyen event'ler burada saklanır.
    Loglama ve audit için kritik — "ne kaybettik?" sorusunu cevaplar.
    """

    def __init__(self, max_size: int = 1000):
        self._queue: deque[OutboxEntry] = deque(maxlen=max_size)

    def add(self, entry: OutboxEntry) -> None:
        entry.status = "failed"
        self._queue.append(entry)
        logger.error(
            f"[DLQ] Event kalici olarak basarisiz: "
            f"gate={entry.gate_id}, id={entry.entry_id}, "
            f"retry={entry.retry_count}"
        )

    @property
    def size(self) -> int:
        return len(self._queue)

    def dump(self) -> list[dict]:
        """Audit için tüm başarısız event'ler."""
        return [
            {"id": e.entry_id, "gate": e.gate_id, "payload": e.payload}
            for e in self._queue
        ]


class EdgeEventPublisher:
    """
    Edge node'dan merkezi sunucuya event iletimi.

    ÖZELLIKLER:
    - Outbox Pattern: network kopsa bile event kaybolmaz
    - Dead Letter Queue: kalıcı hataları yakala
    - Thread-safe: birden fazla thread kullanabilir
    - Mock transport: test için gerçek network gerekmez

    KAVRAM: Dependency Injection (transport)
    Transport fonksiyonu dışarıdan inject edilir.
    Test'te mock_transport, production'da real_transport geçilir.
    """

    def __init__(
        self,
        gate_id: str,
        transport: Optional[Callable[[dict], bool]] = None,
        batch_size: int = 10,
        flush_interval_s: float = 1.0,
    ):
        self._gate_id   = gate_id
        self._transport = transport or self._mock_transport
        self._batch_size = batch_size
        self._flush_interval = flush_interval_s

        # Outbox kuyruğu — thread-safe deque
        self._outbox: deque[OutboxEntry] = deque(maxlen=5000)
        self._dlq = DeadLetterQueue()

        # İstatistikler
        self._sent_count    = 0
        self._failed_count  = 0
        self._total_latency = 0.0

        # Background flush thread
        self._lock     = threading.Lock()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None

        self._entry_counter = 0

        # Dayanıklılık: basit circuit breaker. Ağ sürekli hata verirken
        # merkezi sistemi ve edge CPU'yu retry fırtınasından korur.
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_failure_threshold = 5
        self._circuit_cooldown_s = 2.0
        self._sent_event_ids: set[str] = set()

    def baslat(self) -> None:
        """Background flush thread'i başlat."""
        self._running = True
        self._thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name=f"publisher-{self._gate_id}"
        )
        self._thread.start()
        logger.info(f"[{self._gate_id}] EventPublisher basladi.")

    def durdur(self) -> None:
        """Graceful shutdown — kalan event'leri gönder."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        # Son flush
        self._flush()
        logger.info(
            f"[{self._gate_id}] EventPublisher durduruldu. "
            f"Gonderilen: {self._sent_count}, Basarisiz: {self._failed_count}"
        )

    def publish_frame_result(self, result: FrameSonucu) -> None:
        """
        Frame sonucunu outbox'a ekle.

        Sadece yeni sayılan bagajlar iletilir — her frame değil.
        Bu bant genişliğini önemli ölçüde azaltır.
        """
        if not result.yeni_sayilanlar:
            return  # Yeni tespit yok — iletecek şey yok

        for tespit in result.yeni_sayilanlar:
            self._enqueue(tespit.to_event_dict())

    def publish_alert(self, alert_type: str, data: dict) -> None:
        """Acil alert — öncelikli olarak işlenir."""
        payload = {
            "type":      "alert",
            "alert_type": alert_type,
            "gate_id":   self._gate_id,
            "timestamp": int(time.time() * 1000),
            "data":      data,
        }
        self._enqueue(payload, priority=True)

    def _enqueue(self, payload: dict, priority: bool = False) -> None:
        """Payload'ı outbox'a ekle."""
        with self._lock:
            self._entry_counter += 1
            entry_id = f"{self._gate_id}-{self._entry_counter}"
            payload = dict(payload)
            payload.setdefault("event_id", entry_id)
            payload.setdefault("gate_id", self._gate_id)
            entry = OutboxEntry(
                entry_id=entry_id,
                gate_id=self._gate_id,
                payload=payload,
            )
            if priority:
                self._outbox.appendleft(entry)  # Öncelikli → başa ekle
            else:
                self._outbox.append(entry)

    def _flush_loop(self) -> None:
        """Background thread: periyodik flush."""
        while self._running:
            time.sleep(self._flush_interval)
            self._flush()

    def _flush(self) -> None:
        """
        Outbox'tan event'leri al ve transport ile gönder.
        Başarısız olanları retry, max retry'ı aşanları DLQ'ya at.
        """
        batch = []
        with self._lock:
            while self._outbox and len(batch) < self._batch_size:
                batch.append(self._outbox.popleft())

        if not batch:
            return

        now = time.time()
        if now < self._circuit_open_until:
            # Circuit açıkken event'leri kaybetme: sıraya geri koy.
            with self._lock:
                for entry in reversed(batch):
                    self._outbox.appendleft(entry)
            return

        for entry in batch:
            if entry.entry_id in self._sent_event_ids:
                # Lokal idempotency: aynı entry tekrar sıraya girerse ikinci kez
                # gönderme. Merkezi tarafta da event_id unique constraint olmalıdır.
                continue
            t_start = time.perf_counter()
            success = False
            try:
                success = self._transport(entry.payload)
            except Exception as e:
                logger.warning(
                    f"[{self._gate_id}] Transport hatasi: {e}"
                )

            if success:
                self._sent_event_ids.add(entry.entry_id)
                self._sent_count += 1
                self._consecutive_failures = 0
                self._total_latency += (time.perf_counter() - t_start) * 1000
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._circuit_failure_threshold:
                    self._circuit_open_until = time.time() + self._circuit_cooldown_s
                    logger.warning(
                        f"[{self._gate_id}] Circuit OPEN: "
                        f"{self._consecutive_failures} ardışık hata, "
                        f"{self._circuit_cooldown_s:.1f}s bekleniyor"
                    )
                entry.retry_count += 1
                if entry.is_expired:
                    self._dlq.add(entry)
                    self._failed_count += 1
                else:
                    # Tekrar dene — geri kuyruğa ekle
                    with self._lock:
                        self._outbox.appendleft(entry)

    def _mock_transport(self, payload: dict) -> bool:
        """
        Test transport'u — gerçek network olmadan simüle eder.
        %95 başarı oranı — gerçekçi network davranışı.
        """
        import random
        success = random.random() > 0.05
        if not success:
            logger.debug(
                f"[{self._gate_id}] Mock transport: simulated failure"
            )
        return success

    # ── İstatistikler ──
    @property
    def outbox_size(self) -> int:
        return len(self._outbox)

    @property
    def dlq_size(self) -> int:
        return self._dlq.size

    @property
    def sent_count(self) -> int:
        return self._sent_count

    @property
    def ortalama_latency_ms(self) -> float:
        if self._sent_count == 0:
            return 0.0
        return round(self._total_latency / self._sent_count, 2)

    @property
    def circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until
