# central/models/flight_models.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class UcakTipi(Enum):
    """
    YOL HARİTASI MADDE B2 — Gerçek THY filosu.

    DEĞİŞİKLİK NOTU: Önceki sürümde üçüncü bir "REGIONAL" (ATR) kategorisi
    vardı. Araştırmada THY'nin gerçek filosunda bölgesel/ATR tipi uçak
    olmadığı doğrulandı (filo: A319/A320/A321/A321neo + A330/A350/B777/B787).
    Bu kategori tamamen kaldırıldı.

    SINIRLAMA NOTU (bilinçli, gizlenmiyor): Bu prototipte uçaklar yalnızca
    dar gövde / geniş gövde seviyesinde modellenmektedir. Gerçek ürünleşmede
    kapasite, uçak tipi VE kabin konfigürasyonuna (kuyruk numarası bazlı,
    örnek: A321neo "Airspace XL bin" ile standart A321'den farklı kapasiteye
    sahiptir) göre ayrıştırılmalıdır. Bu basitleştirme bilinçli bir prototip
    kapsamı kararıdır.
    """
    NARROW_BODY = "narrow_body"   # THY: A319 / A320 / A321 / A321neo ailesi
    WIDE_BODY   = "wide_body"     # THY: A330 / A350 / B777 / B787 ailesi

    @property
    def koltuk_sayisi_araligi(self) -> tuple[int, int]:
        """
        Gerçek THY koltuk sayısı aralığı (araştırmayla doğrulanmış).
        Dashboard slider üst sınırı bu değerden türetilir (bkz. Madde C1).
        """
        ranges = {
            UcakTipi.NARROW_BODY: (100, 240),   # A319 (~132) - A321neo (~244)
            UcakTipi.WIDE_BODY:   (250, 480),   # A330 (~280) - B777 (~450)
        }
        return ranges[self]

    @property
    def kapasite(self) -> int:
        """
        Kabin üstü dolap (overhead bin) kapasitesi — yaklaşık bagaj sayısı.

        ARAŞTIRMA NOTU: Overhead bin kapasitesi sanılanın aksine koltuk
        sayısıyla doğrusal büyümüyor. Bulunan kaynaklarda Boeing 777
        (geniş gövde, yüzlerce koltuk) için bin kapasitesi 117-160 roller
        bag aralığında bulundu — yani standart bir dar gövde (737: ~118)
        ile neredeyse aynı seviyede kalabiliyor. Bu yüzden eski "geniş
        gövde = dar gövdenin 2.3 katı" varsayımı (120 vs 280) terk edildi,
        fark daha gerçekçi bir orana çekildi.
        """
        capacities = {
            UcakTipi.NARROW_BODY: 120,   # ~737/A320 standart bin referansı
            UcakTipi.WIDE_BODY:   170,   # 777 örneği (117-160) + pay
        }
        return capacities[self]


class DolulukSeviyesi(Enum):
    NORMAL   = "normal"
    WARNING  = "warning"
    CRITICAL = "critical"


class SistemGuveni(Enum):
    YUKSEK = "yuksek"
    ORTA = "orta"
    DUSUK = "dusuk"


@dataclass(frozen=True)
class UcusBilgisi:
    """
    Check-in sisteminden gelen uçuş verisi.
    frozen=True — uçuş bilgisi değişmez (snapshot).
    """
    ucus_no:           str
    hat:               str        # "IST-DXB"
    ucak_tipi:         UcakTipi
    toplam_yolcu:      int
    cabin_beyan_sayisi: int        # online check-in bagaj beyanı
    oversized_beyan:   int
    gate_id:           str
    kalkis_zamani:     float = field(default_factory=time.time)
    manuel_senaryo:    bool = False  # Dashboard slider ile kurulan manuel/demo senaryo

    @property
    def dolap_kapasitesi(self) -> int:
        return self.ucak_tipi.kapasite

    @property
    def beyan_orani(self) -> float:
        return self.cabin_beyan_sayisi / max(self.toplam_yolcu, 1)


@dataclass
class DolulukTahmini:
    """Prediction Engine çıktısı."""
    ucus_no:               str
    tahmini_doluluk_orani: float
    tahmini_toplam_bagaj:  int      # GEMINI DÜZELTMESİ: kapasiteye bölünmemiş ham sayı
    tahmini_oversized:     int
    kritik_yolcu_sirasi:   Optional[int]
    asim_bekleniyor:       bool
    guven_skoru:           float
    aciklama:              str
    tahmin_metodu:         str   # "rule_based" | "ml_based"
    timestamp:             float = field(default_factory=time.time)


@dataclass
class GateAksiyonu:
    """
    Gate personeline gönderilen operasyonel karar.

    KAVRAM: Command Pattern hafif uygulaması
    Komut nesne olarak temsil edilir — loglanabilir, tekrar oynatılabilir.
    """
    gate_id:        str
    ucus_no:        str
    seviye:         DolulukSeviyesi
    mesaj:          str
    yonlendirme_baslangic: Optional[int]
    gercek_doluluk: float
    tahmini_doluluk: float
    sistem_guveni:  SistemGuveni = SistemGuveni.ORTA
    timestamp:      float = field(default_factory=time.time)
    insan_override: bool  = False
    override_aciklama: Optional[str] = None

    def override_et(self, aciklama: str) -> None:
        """
        Gate personeli sistem kararını ezer.
        Kural 5: İnsan kararı üstündür.
        """
        self.insan_override = True
        self.override_aciklama = aciklama


@dataclass
class FusionCiktisi:
    """FusionEngine'in birleştirilmiş çıktısı."""
    gate_id:         str
    ucus_no:         str
    gercek_doluluk:  float
    tahmini_doluluk: float
    toplam_sayilan:  int
    oversized_sayisi: int
    cabin_ok_sayisi: int
    aksiyon:         GateAksiyonu
    uyari_aktif:     bool
    sistem_guveni:   SistemGuveni = SistemGuveni.ORTA
    timestamp:       float = field(default_factory=time.time)
