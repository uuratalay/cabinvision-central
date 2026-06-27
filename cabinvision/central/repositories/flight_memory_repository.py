# central/repositories/flight_memory_repository.py
#
# MİMARİ KARAR: Repository Pattern
#
# Veri erişim mantığı iş mantığından ayrılır.
# PredictionEngine "TK-1453 hattının geçmişi nedir?" diye sorar,
# nasıl saklandığını (RAM, SQLite, Redis) bilmez.
# Bu soyutlama sayesinde storage değiştirildiğinde
# PredictionEngine hiç değişmez.

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional
import time
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class UcusKaydi:
    """Tamamlanan bir uçuşun özet kaydı."""
    ucus_no:         str
    hat:             str
    toplam_bagaj:    int
    oversized_sayisi: int
    cabin_ok_sayisi:  int
    personal_sayisi:  int
    doluluk_orani:    float
    ucak_tipi:        str
    tamamlanma_zamani: float = field(default_factory=time.time)

    @property
    def oversized_orani(self) -> float:
        return self.oversized_sayisi / max(self.toplam_bagaj, 1)


@dataclass
class HatIstatistigi:
    """
    Bir hat için hesaplanmış istatistikler.

    KAVRAM: Computed Properties
    Değerler saklanmaz, kaydlardan hesaplanır.
    Bu "derive data, don't store it" prensibinin uygulaması.
    """
    hat:              str
    kayit_sayisi:     int
    ort_doluluk:      float
    ort_oversized_oran: float
    std_doluluk:      float
    max_doluluk:      float
    guven_skoru:      float   # Kaç kayda dayalı (az kayıt = düşük güven)


class FlightMemoryRepository:
    """
    Sefer geçmişini saklar ve sorgular.

    ÖZELLIKLER:
    - Hat bazlı istatistik (TK-1453, TK-0090 farklı hat!)
    - Sliding window: son N kayıt tutulur, eski kayıtlar otomatik silinir
    - Thread-safe olmayan (şimdilik) — tek thread kullanımı

    GELECEK: SQLite veya Redis backend eklenebilir.
    Repository interface değişmez, sadece implementasyon değişir.
    """

    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: Her hat için tutulacak maksimum kayıt sayısı
        """
        # hat → deque[UcusKaydi]
        self._kayitlar: dict[str, deque[UcusKaydi]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._window_size = window_size
        self._toplam_kayit = 0

    def kayit_ekle(self, kayit: UcusKaydi) -> None:
        """
        Tamamlanan uçuşu kaydet.
        Hat bazında sliding window'a eklenir.
        """
        self._kayitlar[kayit.hat].append(kayit)
        self._toplam_kayit += 1
        logger.debug(
            f"[FlightMemory] Kayit eklendi: {kayit.ucus_no} "
            f"({kayit.hat}), doluluk={kayit.doluluk_orani:.2f}"
        )

    def hat_istatistigi_al(
        self, hat: str, ucak_tipi: Optional[str] = None
    ) -> Optional[HatIstatistigi]:
        """
        Hat için geçmiş istatistikleri döner.
        Yeterli kayıt yoksa None döner.

        GEMINI 5. TUR MADDE 3 DÜZELTMESİ (kabul edildi):
        Önceki davranış: TÜM geçmiş kayıtlar (uçak tipinden bağımsız)
        karıştırılarak ortalama alınıyordu. Sorun: doluluk_orani zaten
        kapasiteden normalize edilmiş bir YÜZDE olduğu için matematiksel
        olarak "yanlış" değildi, ama KAVRAMSAL bir risk taşıyordu — farklı
        uçak tiplerinin doluluk DİNAMİKLERİ (yolcu profili, bagaj davranışı)
        birbirinden farklı olabilir; aynı hat farklı uçak tipleriyle uçtuysa
        bu farkı görmezden gelmek yanıltıcı tahminlere yol açabilirdi.

        Düzeltme: `ucak_tipi` verilirse, ÖNCE o tipteki kayıtlar filtrelenir.
        Yeterli (>=3) tip-eşleşen kayıt varsa SADECE onlar kullanılır.
        Yeterli kayıt yoksa (yeni bir uçak tipi, az veri) TÜM kayıtlara
        (tip filtresi olmadan) geri dönülür — ama bu durumda güven skoru
        AYRICA düşürülür (tip uyuşmazlığı belirsizliği eklenir).
        """
        tum_kayitlar = list(self._kayitlar.get(hat, []))

        if not tum_kayitlar:
            return None

        tip_uyusmuyor_cezasi = 0.0
        if ucak_tipi is not None:
            tip_filtreli = [k for k in tum_kayitlar if k.ucak_tipi == ucak_tipi]
            if len(tip_filtreli) >= 3:
                kayitlar = tip_filtreli
            else:
                # Yeterli tip-özel veri yok — tüm kayıtlara dön ama güveni düşür
                kayitlar = tum_kayitlar
                tip_uyusmuyor_cezasi = 0.15
                logger.debug(
                    f"[FlightMemory] {hat} hattında '{ucak_tipi}' tipi için "
                    f"yeterli kayıt yok ({len(tip_filtreli)}/3) — tüm "
                    f"kayıtlara dönülüyor, güven cezası uygulanıyor"
                )
        else:
            kayitlar = tum_kayitlar

        doluluklar = [k.doluluk_orani for k in kayitlar]
        oversized_oranlar = [k.oversized_orani for k in kayitlar]

        ort_doluluk = statistics.mean(doluluklar)
        std_doluluk = statistics.stdev(doluluklar) if len(doluluklar) > 1 else 0.0
        ort_over_oran = statistics.mean(oversized_oranlar)

        # Güven skoru: kayıt sayısına göre (az kayıt → düşük güven)
        guven = min(len(kayitlar) / self._window_size, 1.0)
        guven = round(guven * 0.85 + 0.15, 3)  # min 0.15, max 1.0
        guven = round(max(0.05, guven - tip_uyusmuyor_cezasi), 3)

        return HatIstatistigi(
            hat=hat,
            kayit_sayisi=len(kayitlar),
            ort_doluluk=round(ort_doluluk, 3),
            ort_oversized_oran=round(ort_over_oran, 3),
            std_doluluk=round(std_doluluk, 3),
            max_doluluk=round(max(doluluklar), 3),
            guven_skoru=guven,
        )

    def son_n_kayit(self, hat: str, n: int = 5) -> list[UcusKaydi]:
        """Hat için son N kaydı döner. En yeni önce."""
        kayitlar = list(self._kayitlar.get(hat, []))
        return list(reversed(kayitlar[-n:]))

    def yuksek_riskli_hatlar(
        self,
        doluluk_esigi: float = 0.80,
    ) -> list[HatIstatistigi]:
        """
        Ortalama doluluk eşiğini geçen hatları döner.
        Operasyon merkezi için proaktif uyarı.
        """
        riskli = []
        for hat in self._kayitlar:
            ist = self.hat_istatistigi_al(hat)
            if ist and ist.ort_doluluk >= doluluk_esigi:
                riskli.append(ist)
        return sorted(riskli, key=lambda x: x.ort_doluluk, reverse=True)

    @property
    def toplam_kayit_sayisi(self) -> int:
        return self._toplam_kayit

    @property
    def bilinen_hatlar(self) -> list[str]:
        return list(self._kayitlar.keys())
