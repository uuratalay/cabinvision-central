# central/services/fusion_engine.py
#
# FusionEngine: CV sayımı + tahmin → operasyonel karar
# ActionService: kararı ilgili yerlere ilet
# AuditLogService: her şeyi değiştirilemez şekilde logla

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Callable
import time
import json
import logging

from central.models.flight_models import (
    UcusBilgisi, DolulukTahmini, GateAksiyonu,
    FusionCiktisi, DolulukSeviyesi, SistemGuveni
)

logger = logging.getLogger(__name__)

WARNING_THRESHOLD  = 0.75
CRITICAL_THRESHOLD = 0.90


def _sistem_guveni_hesapla(tahmin: DolulukTahmini) -> SistemGuveni:
    """Tahmin güven skorunu dashboard dostu seviyeye çevirir."""
    if tahmin.guven_skoru >= 0.75:
        return SistemGuveni.YUKSEK
    if tahmin.guven_skoru >= 0.50:
        return SistemGuveni.ORTA
    return SistemGuveni.DUSUK


def _seviye_rank(seviye: DolulukSeviyesi) -> int:
    return {
        DolulukSeviyesi.NORMAL: 0,
        DolulukSeviyesi.WARNING: 1,
        DolulukSeviyesi.CRITICAL: 2,
    }[seviye]


# ─────────────────────────────────────────────
# FUSION STRATEJİSİ
# ─────────────────────────────────────────────

class BaseFusionStratejisi(ABC):
    """CV verisi + tahmin → aksiyon üretme stratejisi."""

    @abstractmethod
    def fuse(
        self,
        gercek_oran:  float,
        tahmin:       DolulukTahmini,
        ucus:         UcusBilgisi,
        toplam_sayilan: int,
        oversized_sayisi: int,
    ) -> GateAksiyonu:
        ...


class KuralBazliFusion(BaseFusionStratejisi):
    """
    Kural tabanlı fusion.

    KURAL HİYERARŞİSİ:
    1. Gerçek veri kritik eşiği geçtiyse → her zaman CRITICAL
    2. Tahmin kritik bekliyorsa → CRITICAL (proaktif)
    3. Gerçek veri uyarı eşiğini geçtiyse → WARNING
    4. Aksi halde → NORMAL

    Gerçek veri her zaman tahmine üstündür.
    """

    def fuse(
        self,
        gercek_oran:      float,
        tahmin:           DolulukTahmini,
        ucus:             UcusBilgisi,
        toplam_sayilan:   int,
        oversized_sayisi: int,
    ) -> GateAksiyonu:

        # Seviye belirleme
        # Gerçek oran mevcut demand/capacity oranıdır. Tahmin tarafında ise
        # ham bagaj sayısını kapasiteye bölerek karar motorunu aynı birime
        # taşıyoruz. Böylece yalnızca sabit anlık doluluğa değil, beklenen
        # toplam overhead talebine de bakılır.
        tahmini_talep_orani = min(
            tahmin.tahmini_toplam_bagaj / max(ucus.dolap_kapasitesi, 1),
            1.5,
        )
        karar_orani = max(gercek_oran, tahmini_talep_orani)

        if gercek_oran >= CRITICAL_THRESHOLD:
            seviye = DolulukSeviyesi.CRITICAL
        elif karar_orani >= CRITICAL_THRESHOLD:
            seviye = DolulukSeviyesi.CRITICAL
        elif karar_orani >= WARNING_THRESHOLD or tahmin.asim_bekleniyor:
            seviye = DolulukSeviyesi.WARNING
        else:
            seviye = DolulukSeviyesi.NORMAL

        # Yönlendirme başlangıcı artık yolcu sıra numarası olarak gösterilmez.
        # Canlı kritik durumda geçmiş tahmin sırasına veya bagaj sayısına dayalı
        # Yolcu sıra numarasına bağlı mesajlar canlı kritik durumda yanıltıcı olabiliyordu.
        # Dashboard kararı aksiyon penceresi olarak verir.
        yonlendirme = None

        mesaj = self._mesaj_uret(
            seviye, gercek_oran, tahmin, yonlendirme, oversized_sayisi
        )

        sistem_guveni = _sistem_guveni_hesapla(tahmin)

        return GateAksiyonu(
            gate_id=ucus.gate_id,
            ucus_no=ucus.ucus_no,
            seviye=seviye,
            mesaj=mesaj,
            yonlendirme_baslangic=yonlendirme,
            gercek_doluluk=round(gercek_oran, 3),
            tahmini_doluluk=tahmin.tahmini_doluluk_orani,
            sistem_guveni=sistem_guveni,
        )

    def _mesaj_uret(
        self,
        seviye:       DolulukSeviyesi,
        gercek:       float,
        tahmin:       DolulukTahmini,
        yonlendirme:  Optional[int],
        oversized:    int,
    ) -> str:
        pct = int(gercek * 100)
        if seviye == DolulukSeviyesi.NORMAL:
            return (
                f"NORMAL — Baş üstü dolap talebi %{pct}. "
                f"Boarding devam edebilir."
            )
        if seviye == DolulukSeviyesi.WARNING:
            return (
                f"DİKKAT — Baş üstü dolap talebi %{pct}. "
                f"{oversized} oversized bagaj tespit edildi. "
                f"Boarding akışını yakından izle; gate-check hazırlığı gerekebilir."
            )
        # CRITICAL
        return (
            f"KRİTİK — Baş üstü dolap talebi %{pct}. "
            f"Mevcut boarding grubunda büyük kabin bagajları için "
            f"gate-check hazırlığı başlat."
        )


class FusionEngine:
    """
    CV sayımı ile tahmin katmanını birleştiren orkestratör.

    KAVRAM: Orchestrator Pattern
    FusionEngine iş mantığı üretmez — koordinasyon yapar.
    Her uçuş için internal state tutar (gate durumu).
    """

    def __init__(
        self,
        strateji: BaseFusionStratejisi,
    ):
        self._strateji = strateji
        # gate_id + ucus_no → FusionCiktisi
        self._son_ciktilar: dict[str, FusionCiktisi] = {}

    def guncelle(
        self,
        ucus:             UcusBilgisi,
        tahmin:           DolulukTahmini,
        toplam_sayilan:   int,
        oversized_sayisi: int,
        cabin_ok_sayisi:  int,
    ) -> FusionCiktisi:
        """
        Her yeni bagaj sayıldığında çağrılır.
        Boarding boyunca sürekli güncellenir.
        """
        # Bu oran fiziksel doluluk değil, overhead demand/capacity oranıdır.
        # Kapasite talebi %100'ü aşabilir; dashboard bunu "talep oranı" olarak
        # adlandırmalıdır.
        gercek_oran = toplam_sayilan / max(ucus.dolap_kapasitesi, 1)

        aksiyon = self._strateji.fuse(
            gercek_oran=gercek_oran,
            tahmin=tahmin,
            ucus=ucus,
            toplam_sayilan=toplam_sayilan,
            oversized_sayisi=oversized_sayisi,
        )

        key = f"{ucus.gate_id}-{ucus.ucus_no}"
        onceki = self._son_ciktilar.get(key)
        if onceki and _seviye_rank(onceki.aksiyon.seviye) > _seviye_rank(aksiyon.seviye):
            # Profesyonel alert davranışı: risk seviyesi dashboard'da bir anda
            # aşağı zıplamasın. RESOLVED/ACK ayrı bir kullanıcı aksiyonu olmalıdır.
            aksiyon.seviye = onceki.aksiyon.seviye

        cikti = FusionCiktisi(
            gate_id=ucus.gate_id,
            ucus_no=ucus.ucus_no,
            gercek_doluluk=round(gercek_oran, 3),
            tahmini_doluluk=tahmin.tahmini_doluluk_orani,
            toplam_sayilan=toplam_sayilan,
            oversized_sayisi=oversized_sayisi,
            cabin_ok_sayisi=cabin_ok_sayisi,
            aksiyon=aksiyon,
            uyari_aktif=aksiyon.seviye != DolulukSeviyesi.NORMAL,
            sistem_guveni=aksiyon.sistem_guveni,
        )

        self._son_ciktilar[key] = cikti
        return cikti

    def son_cikti(self, gate_id: str, ucus_no: str) -> Optional[FusionCiktisi]:
        return self._son_ciktilar.get(f"{gate_id}-{ucus_no}")


# ─────────────────────────────────────────────
# ALERT OBSERVER SİSTEMİ
# ─────────────────────────────────────────────

class BaseAlertObserver(ABC):
    """
    KAVRAM: Observer Pattern
    Alert yayınlandığında tüm observer'lar bildirim alır.
    Publisher kimse subscribe ettiğini bilmez.
    """
    @abstractmethod
    def on_alert(self, cikti: FusionCiktisi) -> None:
        ...


class GateDashboardObserver(BaseAlertObserver):
    """Gate personeline alert iletir."""

    def __init__(self, gate_id: str):
        self._gate_id = gate_id
        self._alerts: list[FusionCiktisi] = []

    def on_alert(self, cikti: FusionCiktisi) -> None:
        if cikti.gate_id == self._gate_id:
            self._alerts.append(cikti)
            logger.info(
                f"[GateDashboard-{self._gate_id}] Alert: "
                f"{cikti.aksiyon.seviye.value} — {cikti.aksiyon.mesaj[:60]}"
            )

    @property
    def son_alert(self) -> Optional[FusionCiktisi]:
        return self._alerts[-1] if self._alerts else None


class CentralDashboardObserver(BaseAlertObserver):
    """Operasyon merkezine tüm gate'lerin durumunu iletir."""

    def __init__(self):
        self._gate_durumlari: dict[str, FusionCiktisi] = {}

    def on_alert(self, cikti: FusionCiktisi) -> None:
        self._gate_durumlari[cikti.gate_id] = cikti

    @property
    def aktif_uyarilar(self) -> list[FusionCiktisi]:
        return [c for c in self._gate_durumlari.values() if c.uyari_aktif]

    @property
    def tum_gate_durumlari(self) -> dict[str, FusionCiktisi]:
        return dict(self._gate_durumlari)


class AuditLogObserver(BaseAlertObserver):
    """
    Immutable log — her alert değiştirilemez şekilde kaydedilir.
    Kural 2: Her karar loglanmalı.
    """

    def __init__(self):
        # deque — sonsuz büyümez, eski kayıtlar otomatik silinir
        self._log: deque[dict] = deque(maxlen=10000)

    def on_alert(self, cikti: FusionCiktisi) -> None:
        entry = {
            "timestamp":      cikti.timestamp,
            "gate_id":        cikti.gate_id,
            "ucus_no":        cikti.ucus_no,
            "seviye":         cikti.aksiyon.seviye.value,
            "gercek_doluluk": cikti.gercek_doluluk,
            "tahmin_doluluk": cikti.tahmini_doluluk,
            "toplam_sayilan": cikti.toplam_sayilan,
            "oversized":      cikti.oversized_sayisi,
            "mesaj":          cikti.aksiyon.mesaj,
            "sistem_guveni":  cikti.sistem_guveni.value,
            "override":       cikti.aksiyon.insan_override,
        }
        self._log.append(entry)

    @property
    def log_boyutu(self) -> int:
        return len(self._log)

    def son_n_kayit(self, n: int = 10) -> list[dict]:
        return list(self._log)[-n:]


class ActionService:
    """
    FusionCiktisi'nı observer'lara dağıtan servis.

    KAVRAM: Publisher (Observer Pattern'in yayıncısı)
    Observer'lar subscribe eder, her alert'te bildirim alır.
    ActionService kimin subscribe ettiğini bilmez.
    """

    def __init__(self):
        self._observers: list[BaseAlertObserver] = []

    def subscribe(self, observer: BaseAlertObserver) -> None:
        self._observers.append(observer)
        logger.debug(
            f"Observer eklendi: {type(observer).__name__}"
        )

    def unsubscribe(self, observer: BaseAlertObserver) -> None:
        self._observers = [o for o in self._observers if o is not observer]

    def yayinla(self, cikti: FusionCiktisi) -> None:
        """Tüm observer'lara bildir — polymorphism burada."""
        for observer in self._observers:
            try:
                observer.on_alert(cikti)
            except Exception as e:
                logger.error(
                    f"Observer hatasi ({type(observer).__name__}): {e}"
                )
