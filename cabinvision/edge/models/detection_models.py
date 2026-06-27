# edge/models/detection_models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class BagajTipi(Enum):
    """
    KAVRAM: Enum kullanımı
    Magic string'leri önler. "suitcase" yerine BagajTipi.SUITCASE.
    Tip güvenliği sağlar — geçersiz değer compile time'da değil
    en azından assignment'da yakalanır.
    """
    SUITCASE = "suitcase"
    BACKPACK = "backpack"
    HANDBAG  = "handbag"
    UNKNOWN  = "unknown"

    @classmethod
    def from_coco_id(cls, class_id: int) -> "BagajTipi":
        """COCO class ID → BagajTipi dönüşümü."""
        mapping = {24: cls.BACKPACK, 26: cls.HANDBAG, 28: cls.SUITCASE}
        return mapping.get(class_id, cls.UNKNOWN)


class BoyutSinifi(Enum):
    """
    KAVRAM: Neden UNCERTAIN eklendi?
    Tek RGB kameradan bbox alanı kesin ölçüm değil, zayıf bir göstergedir.
    Eşik sınırına yakın tespitleri zorla bir sınıfa sokmak yerine
    "belirsiz, doğrulama gerekir" diye işaretlemek daha dürüst bir sistemdir.
    Bu sınıf jüriye/kullanıcıya sahte kesinlik göstermemek için var.
    """
    PERSONAL_ITEM = "personal_item"
    CABIN_OK      = "cabin_ok"
    OVERSIZED     = "oversized"
    UNCERTAIN     = "uncertain"   # eşik sınırına yakın, kesin sınıf verilemiyor
    UNKNOWN       = "unknown"     # kalibrasyon geçersiz / hesaplanamadı


@dataclass
class BoundingBox:
    """
    Bounding box değer nesnesi.

    KAVRAM: Value Object
    Kimliği yoktur, sadece değeri vardır. İki BoundingBox aynı
    koordinatlara sahipse eşittir.
    frozen=True bu semantiği destekler.
    """
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"Gecersiz bbox: ({self.x1},{self.y1})->({self.x2},{self.y2})"
            )

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union — iki bbox örtüşme oranı."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / max(union, 1)


@dataclass
class BagajTespiti:
    """
    Tek bir bagaj tespitinin tam veri yapısı.

    KAVRAM: Nested dataclass
    Metadata iç sınıf olarak tanımlanır — sadece BagajTespiti'ne aittir.
    Dışarıdan erişilebilir ama semantik olarak "bu tespite ait meta" mesajı verir.
    """
    track_id:     int
    bagaj_tipi:   BagajTipi
    boyut_sinifi: BoyutSinifi
    bbox:         BoundingBox
    confidence:   float
    frame_no:     int
    gate_id:      str
    sayildi_mi:   bool = False

    @dataclass
    class Metadata:
        """Teknik meta — loglama ve audit için."""
        kalibrasyon_version: str
        model_version:       str
        inference_ms:        float
        timestamp_ms:        int

    metadata: Optional[Metadata] = None

    def to_event_dict(self) -> dict:
        """EventPublisher için serileştirme."""
        return {
            "track_id":     self.track_id,
            "gate_id":      self.gate_id,
            "bagaj_tipi":   self.bagaj_tipi.value,
            "boyut_sinifi": self.boyut_sinifi.value,
            "bbox":         self.bbox.to_tuple(),
            "confidence":   round(self.confidence, 3),
            "frame_no":     self.frame_no,
            "sayildi_mi":   self.sayildi_mi,
            "meta": {
                "kalibrasyon_v": self.metadata.kalibrasyon_version if self.metadata else "unknown",
                "model_v":       self.metadata.model_version if self.metadata else "unknown",
                "inference_ms":  self.metadata.inference_ms if self.metadata else 0,
                "timestamp_ms":  self.metadata.timestamp_ms if self.metadata else 0,
            }
        }


@dataclass
class FrameSonucu:
    """Bir frame'in tüm tespit sonuçları."""
    frame_no:   int
    gate_id:    str
    tespitler:  list[BagajTespiti] = field(default_factory=list)
    isleme_ms:  float = 0.0
    yeni_sayilanlar: list[BagajTespiti] = field(default_factory=list)

    @property
    def oversized_sayisi(self) -> int:
        return sum(
            1 for t in self.yeni_sayilanlar
            if t.boyut_sinifi == BoyutSinifi.OVERSIZED
        )

    @property
    def cabin_ok_sayisi(self) -> int:
        return sum(
            1 for t in self.yeni_sayilanlar
            if t.boyut_sinifi == BoyutSinifi.CABIN_OK
        )
