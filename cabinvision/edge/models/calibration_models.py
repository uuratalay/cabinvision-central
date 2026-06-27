# edge/models/calibration_models.py
#
# KAVRAM: Neden ayrı bir models dosyası?
#
# Veri yapıları (ne taşındığı) ile iş mantığı (nasıl işlendiği) birbirinden
# ayrı tutulur. Bu "Separation of Concerns" prensibinin somut uygulaması.
# CalibrationService bu modelleri kullanır ama sahip olmaz.
# Test yazarken de sadece bu dosyayı import etmek yeterli olur.
#
# KAVRAM: @dataclass vs normal class
# Normal class'ta __init__, __repr__, __eq__ metodlarını elle yazmak gerekir.
# @dataclass bunları otomatik üretir. Sadece "veri tutan" yapılar için idealdir.
# İş mantığı içeren class'lar normal class olarak yazılır.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class GatePhysicalParams:
    """
    Gate'in fiziksel parametreleri.

    NEDEN frozen=True?
    Kalibrasyon parametreleri bir kez belirlendikten sonra değişmemeli.
    frozen=True bu nesneyi immutable (değiştirilemez) yapar.
    Yanlışlıkla params.camera_height = 5.0 denmesini engeller.
    Bu "defensive programming" prensibidir.

    Attributes:
        gate_id: Gate'in benzersiz kimliği (ör: "IST-GATE-12")
        camera_height_m: Kameranın yerden yüksekliği (metre)
        camera_tilt_deg: Kameranın yere göre açısı (derece, 0=yere dik)
        camera_fov_horizontal_deg: Yatay görüş açısı
        camera_fov_vertical_deg: Dikey görüş açısı
        frame_width_px: Frame genişliği (piksel)
        frame_height_px: Frame yüksekliği (piksel)
    """
    gate_id: str
    camera_height_m: float
    camera_tilt_deg: float
    camera_fov_horizontal_deg: float
    camera_fov_vertical_deg: float
    frame_width_px: int
    frame_height_px: int

    def __post_init__(self):
        """
        KAVRAM: __post_init__
        @dataclass'ın __init__'i çalıştıktan sonra çağrılır.
        Validation için ideal yer — nesne yaratılmadan önce kural kontrolleri.
        """
        if self.camera_height_m <= 0:
            raise ValueError(
                f"Kamera yuksekligi pozitif olmali: {self.camera_height_m}"
            )
        if not (0 <= self.camera_tilt_deg <= 90):
            raise ValueError(
                f"Kamera acisi 0-90 derece arasinda olmali: {self.camera_tilt_deg}"
            )
        if self.frame_width_px <= 0 or self.frame_height_px <= 0:
            raise ValueError("Frame boyutlari pozitif olmali")


@dataclass(frozen=True)
class ReferenceObject:
    """
    Kalibrasyon için kullanılan referans nesne.

    Manuel kalibrasyonda bir referans nesne (ör: standart kabin bagajı)
    gate'e tutulur. Bu nesnenin bilinen gerçek boyutları ile kameradaki
    piksel boyutu eşleştirilerek dönüşüm faktörü hesaplanır.

    Attributes:
        real_width_cm: Gerçek genişlik (cm)
        real_height_cm: Gerçek yükseklik (cm)
        pixel_bbox: Referans nesnenin frame'deki bbox (x1, y1, x2, y2)
        distance_from_camera_m: Kameradan uzaklık (metre)
    """
    real_width_cm: float
    real_height_cm: float
    pixel_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    distance_from_camera_m: float

    @property
    def pixel_width(self) -> int:
        return self.pixel_bbox[2] - self.pixel_bbox[0]

    @property
    def pixel_height(self) -> int:
        return self.pixel_bbox[3] - self.pixel_bbox[1]

    @property
    def pixel_area(self) -> int:
        return self.pixel_width * self.pixel_height

    @property
    def real_area_cm2(self) -> float:
        return self.real_width_cm * self.real_height_cm


@dataclass
class CalibrationResult:
    """
    Kalibrasyon işleminin çıktısı.

    NEDEN frozen=False?
    Kalibrasyon sonucu sonradan güncellenebilir (recalibration).
    Ayrıca is_valid flag'i runtime'da değişebilir.

    Attributes:
        gate_id: Hangi gate için kalibrasyon yapıldı
        version: Kalibrasyon versiyonu (model değişince artırılır)
        cm_per_pixel_at_ref_distance: Referans mesafede cm/piksel oranı
        size_thresholds_px: Boyut sınıflandırma eşikleri (piksel alanı)
        homography_matrix: Perspektif düzeltme matrisi (opsiyonel)
        calibration_method: "manual" veya "auto"
        is_valid: Kalibrasyon geçerli mi
        confidence_score: Kalibrasyonun güven skoru (0.0-1.0)
    """
    gate_id: str
    version: str
    cm_per_pixel_at_ref_distance: float
    size_thresholds_px: dict[str, float]
    homography_matrix: Optional[np.ndarray]
    calibration_method: str
    is_valid: bool = True
    confidence_score: float = 1.0
    notes: str = ""

    def classify_bbox(self, bbox: tuple[int, int, int, int]) -> str:
        """
        Bbox alanına göre boyut sınıfı döner.

        Bu metodun CalibrationResult içinde olması kasıtlı.
        "Sınıflandırma kalibrasyona bağlıdır" — kalibrasyon değişince
        sınıflandırma otomatik değişir. Başka bir yerde bu mantık olsaydı
        kalibrasyon güncellenince orayı da güncellemek gerekirdi.

        KAVRAM: High Cohesion — ilgili şeyler bir arada durur.

        BELİRSİZLİK BANDI:
        Tek RGB kameradan bbox alanı bagajın gerçek boyutunu kesin
        temsil etmez (kamera açısı, bagajın duruşu, hangi yüzeyin
        göründüğü değişkendir). Bu yüzden eşik sınırlarına çok yakın
        (±%15 içinde) alanlar zorla bir sınıfa sokulmaz, "uncertain"
        olarak işaretlenir. Bu, sahte kesinlik üretmemek için bilinçli
        bir tasarım kararıdır — bkz. SINIRDA_BELIRSIZLIK_ORANI.
        """
        if not self.is_valid:
            raise RuntimeError(
                f"Gate {self.gate_id} kalibrasyonu gecersiz, "
                "siniflandirma yapilamaz"
            )

        x1, y1, x2, y2 = bbox
        alan = (x2 - x1) * (y2 - y1)

        personal_max = self.size_thresholds_px["personal_item_max"]
        cabin_ok_max = self.size_thresholds_px["cabin_ok_max"]

        # Belirsizlik marjı — eşik etrafında ±%15'lik bant
        marj = 0.15
        personal_alt = personal_max * (1 - marj)
        personal_ust = personal_max * (1 + marj)
        cabin_alt = cabin_ok_max * (1 - marj)
        cabin_ust = cabin_ok_max * (1 + marj)

        # Net bölgeler
        if alan < personal_alt:
            return "personal_item"
        if personal_ust < alan < cabin_alt:
            return "cabin_ok"
        if alan > cabin_ust:
            return "oversized"

        # Sınır bantları — kesin sınıf verilemez
        return "uncertain"

    def to_dict(self) -> dict:
        """Serileştirme için — network üzerinden gönderilecek."""
        return {
            "gate_id": self.gate_id,
            "version": self.version,
            "cm_per_pixel": self.cm_per_pixel_at_ref_distance,
            "thresholds": self.size_thresholds_px,
            "method": self.calibration_method,
            "is_valid": self.is_valid,
            "confidence": self.confidence_score,
        }
