# central/services/prediction_engine.py
#
# MİMARİ KARAR: Strategy Pattern (tekrar)
# Şimdi kural bazlı çalışır. Yeterli veri birikince ML stratejisine geçiş.
# PredictionEngine hangi stratejiyi kullandığını bilmez.
#
# SEFER HAFIZASI entegrasyonu:
# Tahminin ana omurgası sefer hafızası + uçak kapasitesidir.
# Check-in/PNR tarafındaki kabin bagaj sinyali doğrulanmadığı için ana
# belirleyici değildir; varsa düşük ağırlıklı opsiyonel feature olarak kullanılır.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import math
import logging

from central.models.flight_models import (
    UcusBilgisi, DolulukTahmini, DolulukSeviyesi
)
from central.repositories.flight_memory_repository import (
    FlightMemoryRepository, HatIstatistigi
)

logger = logging.getLogger(__name__)

# Alert eşikleri
WARNING_THRESHOLD  = 0.75
CRITICAL_THRESHOLD = 0.90

# Prototip konfigürasyonu — gizli karar sabiti değil, açık/kalibre edilebilir değerler.
# Gerçek operasyonel veri geldiğinde bu değerler backtest ile ayarlanmalıdır.
VARSAYILAN_BAGAJ_DOLULUK_ORANI = 0.55
GECMIS_MAX_AGIRLIK = 0.70
GECMIS_HEDEF_KAYIT = 50
PNR_OPSIYONEL_AGIRLIK = 0.15
MANUEL_SENARYO_PNR_AGIRLIK = 0.70
MANUEL_SENARYO_MAX_GECMIS_AGIRLIK = 0.15
MIN_TIP_ESLESEN_KAYIT = 3


class BaseTahminStratejisi(ABC):
    """
    Tüm tahmin stratejilerinin soyut temeli.
    Override zorunlu: tahmin_uret()
    """

    @abstractmethod
    def tahmin_uret(
        self,
        ucus: UcusBilgisi,
        hat_istatistigi: Optional[HatIstatistigi],
    ) -> DolulukTahmini:
        ...

    def _kritik_sira_hesapla(
        self,
        kapasite: int,
        toplam_yolcu: int,
        doluluk_orani: float,
    ) -> Optional[int]:
        """Kapasitenin dolacağı yolcu sırası."""
        if doluluk_orani < WARNING_THRESHOLD:
            return None
        if toplam_yolcu == 0 or doluluk_orani == 0:
            return None
        sira = int(kapasite / (doluluk_orani * toplam_yolcu) * toplam_yolcu)
        return min(max(sira, 1), toplam_yolcu)

    def _guven_hesapla(
        self,
        ucus: UcusBilgisi,
        hat_istatistigi: Optional[HatIstatistigi],
    ) -> float:
        """
        Tahminin güven skoru.

        Check-in beyanı artık ana güven kaynağı değildir; çünkü kabin bagajı
        beyanının gerçek sistemde bulunup bulunmadığı doğrulanmamıştır. Güven
        öncelikle sefer hafızası kayıt sayısı/güven skorundan gelir.
        """
        guven = 0.35  # düşük ama sıfır olmayan prototip taban güveni
        if hat_istatistigi:
            guven += hat_istatistigi.guven_skoru * 0.50
        if ucus.cabin_beyan_sayisi > 0:
            guven += 0.05  # opsiyonel PNR/beyan sinyali varlığı
        return round(min(guven, 0.95), 3)

    def _gecmis_agirlik_hesapla(
        self,
        hat_istatistigi: Optional[HatIstatistigi],
    ) -> float:
        """Kayıt sayısına göre açıklanabilir geçmiş ağırlığı."""
        if not hat_istatistigi or hat_istatistigi.kayit_sayisi < MIN_TIP_ESLESEN_KAYIT:
            return 0.0
        return round(
            min(
                GECMIS_MAX_AGIRLIK,
                (hat_istatistigi.kayit_sayisi / GECMIS_HEDEF_KAYIT) * GECMIS_MAX_AGIRLIK,
            ),
            3,
        )


class KuralBazliStrateji(BaseTahminStratejisi):
    """
    Kural tabanlı tahmin.

    Formül:
    - Ana sinyal: hat/ucak tipi sefer hafızası
    - Fallback: açıkça işaretli prototip varsayımı
    - Opsiyonel sinyal: doğrulanmamış check-in/PNR beyanı düşük ağırlıkla

    Eski `%40 beyan etmeyen getirir` sabiti kaldırıldı; çünkü doğrulanmamış
    veriyi ana karar girdisi yapıyordu.
    """

    def tahmin_uret(
        self,
        ucus: UcusBilgisi,
        hat_istatistigi: Optional[HatIstatistigi],
    ) -> DolulukTahmini:

        kapasite = ucus.dolap_kapasitesi

        # 1) Fallback: veri yoksa açıkça işaretli prototip varsayımı.
        varsayilan_doluluk = VARSAYILAN_BAGAJ_DOLULUK_ORANI

        # 2) Ana sinyal: sefer hafızası. Kayıt sayısına göre dinamik ağırlık.
        gecmis_agirlik = self._gecmis_agirlik_hesapla(hat_istatistigi)
        gecmis_doluluk = (
            hat_istatistigi.ort_doluluk
            if hat_istatistigi and hat_istatistigi.kayit_sayisi >= MIN_TIP_ESLESEN_KAYIT
            else varsayilan_doluluk
        )

        # 3) Opsiyonel PNR/check-in sinyali.
        # Bu sinyal kabin bagajı beyanı olarak doğrulanmadığı için yalnızca
        # düşük ağırlıklı yardımcı feature'dır; `%40 beyan etmeyen` varsayımı
        # tamamen kaldırılmıştır.
        if getattr(ucus, "manuel_senaryo", False):
            # Manuel dashboard senaryosunda slider kullanıcının kurduğu risk
            # senaryosudur; bu yüzden PNR/ön tahmin sinyali konservatif
            # operasyon moduna göre daha güçlü yansıtılır. Yine de geçmiş veri
            # tamamen atılmaz, düşük tavanla bağlam sinyali olarak kalır.
            gecmis_agirlik = min(gecmis_agirlik, MANUEL_SENARYO_MAX_GECMIS_AGIRLIK)
            pnr_agirlik = MANUEL_SENARYO_PNR_AGIRLIK if ucus.cabin_beyan_sayisi > 0 else 0.0
        else:
            pnr_agirlik = PNR_OPSIYONEL_AGIRLIK if ucus.cabin_beyan_sayisi > 0 else 0.0

        pnr_doluluk = min(ucus.cabin_beyan_sayisi / max(kapasite, 1), 1.0)

        varsayilan_agirlik = max(0.0, 1.0 - gecmis_agirlik - pnr_agirlik)
        doluluk = (
            varsayilan_doluluk * varsayilan_agirlik
            + gecmis_doluluk * gecmis_agirlik
            + pnr_doluluk * pnr_agirlik
        )

        logger.debug(
            f"[{ucus.ucus_no}] Tahmin harmanlama: "
            f"varsayilan={varsayilan_doluluk:.2f}*{varsayilan_agirlik:.2f}, "
            f"gecmis={gecmis_doluluk:.2f}*{gecmis_agirlik:.2f}, "
            f"pnr={pnr_doluluk:.2f}*{pnr_agirlik:.2f}"
        )

        doluluk = round(min(doluluk, 1.0), 3)
        asim    = doluluk >= CRITICAL_THRESHOLD
        guven   = self._guven_hesapla(ucus, hat_istatistigi)

        kritik_sira = self._kritik_sira_hesapla(
            ucus.dolap_kapasitesi, ucus.toplam_yolcu, doluluk
        )

        aciklama = self._aciklama(doluluk, kritik_sira, asim, ucus)

        # GEMINI DÜZELTMESİ (3. tur, madde 1) — kritik kenar durumu (edge case):
        # Sefer hafızası ağırlığı yüksekken (bkz. Madde A5) ve yolcu sayısı
        # DÜŞÜK, kapasite YÜKSEK olduğunda (örnek: wide body kapasite=400,
        # yolcu=30, geçmiş ortalama=%80), düzeltilmemiş hesap şunu üretir:
        # ceil(0.80*400)=320 bagaj — ama uçakta sadece 30 yolcu var, 320
        # bagaj fiziksel olarak imkansız. Bu yüzden tahmin edilen toplam
        # bagaj sayısı, mevcut yolcu sayısını AŞAMAZ (clamping).
        tahmini_toplam_bagaj = min(
            math.ceil(doluluk * ucus.dolap_kapasitesi),
            ucus.toplam_yolcu
        )

        return DolulukTahmini(
            ucus_no=ucus.ucus_no,
            tahmini_doluluk_orani=doluluk,
            tahmini_toplam_bagaj=tahmini_toplam_bagaj,
            tahmini_oversized=ucus.oversized_beyan,
            kritik_yolcu_sirasi=kritik_sira,
            asim_bekleniyor=asim,
            guven_skoru=guven,
            aciklama=aciklama,
            tahmin_metodu="rule_based",
        )

    def _aciklama(
        self,
        doluluk: float,
        kritik_sira: Optional[int],
        asim: bool,
        ucus: UcusBilgisi,
    ) -> str:
        pct = int(doluluk * 100)
        if doluluk < WARNING_THRESHOLD:
            return (
                f"Tahmini baş üstü dolap talebi %{pct} — normal seviye. "
                f"Boarding devam edebilir."
            )
        if doluluk < CRITICAL_THRESHOLD:
            return (
                f"Tahmini baş üstü dolap talebi %{pct} — UYARI. "
                f"Mevcut boarding grubunda büyük kabin bagajları için "
                f"gate-check hazırlığı planla."
            )
        return (
            f"Tahmini baş üstü dolap talebi %{pct} — KAPASİTE AŞIMI RİSKİ. "
            f"Boarding başlamadan gate-check hazırlığı planla."
        )


class MLBazliStrateji(BaseTahminStratejisi):
    """
    ML tabanlı tahmin — gelecek iterasyon için yer tutucu.

    Şu an KuralBazliStrateji'ye fallback yapar.
    Yeterli geçmiş veri birikince (>100 kayıt) gerçek ML modeli buraya.

    KAVRAM: Null Object Pattern hafif uygulaması
    Henüz implement edilmemiş ama sistemi bozmayan bir strateji.
    """

    def __init__(self):
        self._fallback = KuralBazliStrateji()
        logger.warning(
            "MLBazliStrateji henuz implement edilmedi. "
            "KuralBazliStrateji'ye fallback yapiliyor."
        )

    def tahmin_uret(
        self,
        ucus: UcusBilgisi,
        hat_istatistigi: Optional[HatIstatistigi],
    ) -> DolulukTahmini:
        result = self._fallback.tahmin_uret(ucus, hat_istatistigi)
        result.tahmin_metodu = "ml_based_fallback"
        return result


class PredictionEngine:
    """
    Boarding başlamadan baş üstü dolap talebi tahmini üretir.

    BAĞIMLILIKLAR (Dependency Injection):
    - BaseTahminStratejisi: hangi algoritma?
    - FlightMemoryRepository: geçmiş veriye eriş

    NOT: Engine uçuş bilgisini OKUR, yazmaz.
    Uçuş tamamlandığında FlightMemoryRepository'ye kayıt
    AuditLogService üzerinden yapılır.
    """

    def __init__(
        self,
        strateji: BaseTahminStratejisi,
        memory: FlightMemoryRepository,
    ):
        self._strateji = strateji
        self._memory   = memory
        self._tahmin_cache: dict[str, DolulukTahmini] = {}

    @classmethod
    def create(
        cls,
        metod: str = "rule_based",
        memory: Optional[FlightMemoryRepository] = None,
    ) -> "PredictionEngine":
        """Factory method."""
        mem = memory or FlightMemoryRepository()
        stratejiler = {
            "rule_based": KuralBazliStrateji,
            "ml_based":   MLBazliStrateji,
        }
        if metod not in stratejiler:
            raise ValueError(f"Bilinmeyen metod: {metod}")
        return cls(strateji=stratejiler[metod](), memory=mem)

    def tahmin_uret(self, ucus: UcusBilgisi) -> DolulukTahmini:
        """
        Uçuş için baş üstü dolap talebi tahmini üretir.

        Cache: Aynı uçuş için tekrar tahmin üretilmez.
        Uçuş bilgisi değişmediği sürece tahmin de değişmez.
        """
        if ucus.ucus_no in self._tahmin_cache:
            logger.debug(
                f"[{ucus.ucus_no}] Tahmin cache'den donuluyor."
            )
            return self._tahmin_cache[ucus.ucus_no]

        # GEMINI 5. TUR MADDE 3 DÜZELTMESİ: uçak tipi de geçiriliyor —
        # sefer hafızası artık SADECE aynı uçak tipindeki geçmiş kayıtları
        # önceliklendiriyor (yeterli veri varsa), tip-bağımsız karıştırma
        # riski azaltılıyor.
        hat_ist = self._memory.hat_istatistigi_al(ucus.hat, ucus.ucak_tipi.value)

        tahmin = self._strateji.tahmin_uret(ucus, hat_ist)
        self._tahmin_cache[ucus.ucus_no] = tahmin

        logger.info(
            f"[{ucus.ucus_no}] Tahmin: bas_ustu_talep=%"
            f"{int(tahmin.tahmini_doluluk_orani*100)}, "
            f"asim={tahmin.asim_bekleniyor}, "
            f"guven={tahmin.guven_skoru:.2f}, "
            f"metod={tahmin.tahmin_metodu}"
        )
        return tahmin

    def tahmin_gecersiz_kil(self, ucus_no: str) -> None:
        """Uçuş verisi değişince cache'i temizle."""
        self._tahmin_cache.pop(ucus_no, None)

    def strateji_degistir(self, metod: str) -> None:
        """Runtime strateji değişimi."""
        stratejiler = {
            "rule_based": KuralBazliStrateji,
            "ml_based":   MLBazliStrateji,
        }
        if metod not in stratejiler:
            raise ValueError(f"Bilinmeyen metod: {metod}")
        self._strateji = stratejiler[metod]()
        self._tahmin_cache.clear()
        logger.info(f"PredictionEngine stratejisi degisti: {metod}")
