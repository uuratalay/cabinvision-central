# edge/services/inference_service.py
#
# MİMARİ KARAR: Adapter Pattern (YOLOv8 için)
# YOLOv8'in API'si değişebilir (v8 → v9 → v10).
# BaseDetector soyutlaması sayesinde sadece adapter değişir,
# InferenceService hiç dokunulmaz.
#
# MİMARİ KARAR: Pipeline Pattern
# process_frame() → detect → classify → track → filter → output
# Her adım bağımsız, test edilebilir, değiştirilebilir.
#
# PERFORMANS / GÜVENİLİRLİK: ConfidencePolicy
# Önceki sürümde threshold son N frame'in ortalama confidence değerine göre
# gizlice değişiyordu. Bu davranış düşük ışık/kalabalık sahnede yanlış
# pozitifleri artırabildiği için kaldırıldı. Yeni yapı threshold'u sabit tutar,
# sahne kalitesi bozulursa bunu ayrı bir quality flag olarak raporlar.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from collections import deque
import time
import random
import logging
import numpy as np

from edge.models.detection_models import (
    BagajTespiti, BagajTipi, BoyutSinifi,
    BoundingBox, FrameSonucu,
)
from edge.models.calibration_models import CalibrationResult

logger = logging.getLogger(__name__)

MODEL_VERSION = "yolov8n-cabinvision-v1.0"


# ─────────────────────────────────────────────
# DETECTOR SOYUTLAMASI
# ─────────────────────────────────────────────

class BaseDetector(ABC):
    """
    Nesne tespit modellerinin soyut arayüzü.

    KAVRAM: Adapter Pattern (Target Interface)
    InferenceService bu interface ile konuşur.
    YOLOv8Detector, MockDetector vb. bu interface'i implement eder.
    Model değişince sadece concrete class değişir.
    """

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
    ) -> list[dict]:
        """
        Ham tespitler döner.
        Her tespit: {class_id, confidence, bbox: (x1,y1,x2,y2)}
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Model ısınma — ilk inference'ın gecikmesini önler."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Model yüklendi ve kullanıma hazır mı?"""
        ...


class MockDetector(BaseDetector):
    """
    Test ve geliştirme için mock dedektör.

    Gerçek YOLOv8 olmadan sistemi test etmek için.
    Gerçekçi bagaj tespitleri simüle eder.
    Seed ile deterministik — testlerde tutarlı sonuç.
    """

    COCO_BAGAJ_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
    FRAME_W, FRAME_H = 1280, 720

    def __init__(self, seed: int = 42, avg_detections: int = 3):
        self._rng = random.Random(seed)
        self._avg_detections = avg_detections
        self._ready = False
        self._call_count = 0

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
    ) -> list[dict]:
        if not self._ready:
            raise RuntimeError("warmup() cagrilmadi")

        self._call_count += 1
        n = max(0, int(self._rng.gauss(self._avg_detections, 1)))
        results = []

        for _ in range(n):
            class_id = self._rng.choice(self.COCO_BAGAJ_CLASSES)

            # Sınıfa göre gerçekçi boyut dağılımı
            if class_id == 28:  # suitcase — büyük
                w = self._rng.randint(80, 200)
                h = self._rng.randint(70, 180)
            elif class_id == 24:  # backpack — orta
                w = self._rng.randint(50, 120)
                h = self._rng.randint(60, 140)
            else:  # handbag — küçük
                w = self._rng.randint(30, 80)
                h = self._rng.randint(25, 70)

            x1 = self._rng.randint(50, self.FRAME_W - w - 50)
            y1 = self._rng.randint(100, self.FRAME_H - h - 50)
            conf = self._rng.uniform(confidence_threshold + 0.05, 0.97)

            results.append({
                "class_id":  class_id,
                "confidence": round(conf, 3),
                "bbox":       (x1, y1, x1 + w, y1 + h),
            })

        return results

    def warmup(self) -> None:
        self._ready = True
        logger.info("[MockDetector] Isindi. Hazir.")

    @property
    def is_ready(self) -> bool:
        return self._ready


class YOLOv8Detector(BaseDetector):
    """
    Gerçek YOLOv8 dedektörü.

    KAVRAM: Lazy Loading
    __init__'te model yüklenmez — yükleme 2-3 saniye sürer.
    warmup() çağrıldığında yüklenir. Bu "ihtiyaç anında yükle" prensibi.
    """

    TARGET_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase

    def __init__(self, model_path: str = "yolov8n.pt"):
        self._model_path = model_path
        self._model = None
        self._ready = False

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
    ) -> list[dict]:
        if not self._ready:
            raise RuntimeError(
                "YOLOv8 hazir degil. warmup() cagrisi gerekiyor."
            )

        results = self._model(
            frame,
            conf=confidence_threshold,
            classes=self.TARGET_CLASSES,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id":   int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox":       tuple(map(int, box.xyxy[0].tolist())),
                })
        return detections

    def warmup(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            # Dummy frame ile ısınma
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            self._ready = True
            logger.info(f"[YOLOv8] Model yuklendi: {self._model_path}")
        except ImportError:
            raise ImportError(
                "ultralytics kurulu degil. "
                "pip install ultralytics"
            )

    @property
    def is_ready(self) -> bool:
        return self._ready


# ─────────────────────────────────────────────
# TRACKER SOYUTLAMASI
# ─────────────────────────────────────────────

class BaseTracker(ABC):
    """ByteTrack ve diğer tracker'lar için soyut arayüz."""

    @abstractmethod
    def update(self, detections: list[dict]) -> list[dict]:
        """
        Tespitleri tracker'a besle, track_id atanmış tespitler al.
        Her tespit: {..., track_id: int}
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Yeni uçuş için tracker'ı sıfırla."""
        ...


class SimpleTracker(BaseTracker):
    """
    Basit IoU tabanlı tracker.

    ByteTrack olmadan çalışır — prototype ve test için.
    Son frame'deki bbox'larla mevcut tespitleri IoU ile eşleştirir.
    IoU > threshold ise aynı nesne kabul eder.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 5):
        self._tracks: dict[int, dict] = {}  # track_id → track info
        self._next_id = 1
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost_frames
        self._frame_count = 0

    def update(self, detections: list[dict]) -> list[dict]:
        self._frame_count += 1

        if not detections:
            self._age_tracks()
            return []

        # Mevcut track'lerle eşleştir
        assigned = set()
        results = []

        for det in detections:
            best_track_id = None
            best_iou = self._iou_threshold

            det_bbox = BoundingBox(*det["bbox"])

            for tid, track in self._tracks.items():
                if tid in assigned:
                    continue
                track_bbox = BoundingBox(*track["bbox"])
                iou = det_bbox.iou(track_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None:
                # Mevcut track güncelle
                self._tracks[best_track_id].update({
                    "bbox":        det["bbox"],
                    "confidence":  det["confidence"],
                    "lost_frames": 0,
                    "last_frame":  self._frame_count,
                })
                assigned.add(best_track_id)
                det["track_id"] = best_track_id
            else:
                # Yeni track oluştur
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox":        det["bbox"],
                    "confidence":  det["confidence"],
                    "class_id":    det["class_id"],
                    "lost_frames": 0,
                    "last_frame":  self._frame_count,
                    "first_frame": self._frame_count,
                }
                det["track_id"] = tid

            results.append(det)

        self._age_tracks()
        return results

    def _age_tracks(self) -> None:
        """Uzun süredir görülmeyen track'leri sil."""
        to_delete = []
        for tid, track in self._tracks.items():
            if track["last_frame"] < self._frame_count:
                track["lost_frames"] += 1
                if track["lost_frames"] > self._max_lost:
                    to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0


# ─────────────────────────────────────────────
# ADAPTIVE THRESHOLD
# ─────────────────────────────────────────────

class AdaptiveThreshold:
    """
    Geriye dönük uyumlu sabit confidence policy.

    PROFESYONEL DÜZENLEME:
    Eski formül `base + (mean_conf - 0.65) * 0.15` veriye dayanmayan iki
    sabit içeriyor ve model zorlandığında eşiği düşürerek yanlış pozitif
    riskini artırıyordu. Bu sınıfın adı dış API'yi kırmamak için korundu;
    ancak artık threshold değerini otomatik değiştirmez. Son N frame'in
    confidence istatistiği yalnızca sahne/girdi kalitesini işaretlemek için
    kullanılır.
    """

    LOW_CONFIDENCE_MEAN = 0.50  # prototip kalite bayrağı; pilot veriyle ayarlanmalı

    def __init__(
        self,
        base_threshold: float = 0.50,
        window_size: int = 100,
        min_threshold: float = 0.30,
        max_threshold: float = 0.70,
    ):
        if not 0.0 <= base_threshold <= 1.0:
            raise ValueError("base_threshold 0.0-1.0 aralığında olmalıdır")
        self._base = base_threshold
        self._window = deque(maxlen=window_size)
        # Geriye dönük uyumluluk: min/max parametreleri kabul edilir ama
        # threshold clamp/oynatma için kullanılmaz.
        self._min = min_threshold
        self._max = max_threshold
        self._quality_flag = "UNKNOWN"

    def update(self, confidences: list[float]) -> None:
        """Yeni frame confidence değerleriyle kalite penceresini güncelle."""
        self._window.extend(confidences)
        if not self._window:
            self._quality_flag = "UNKNOWN"
            return

        mean_conf = sum(self._window) / len(self._window)
        self._quality_flag = (
            "LOW_CONFIDENCE" if mean_conf < self.LOW_CONFIDENCE_MEAN else "OK"
        )

    @property
    def value(self) -> float:
        """Detector'a gönderilen sabit threshold."""
        return round(self._base, 3)

    @property
    def quality_flag(self) -> str:
        return self._quality_flag


# ─────────────────────────────────────────────
# INFERENCE SERVICE
# ─────────────────────────────────────────────

class InferenceService:
    """
    Ana CV inference servisi.

    SORUMLULUK (Single Responsibility):
    Kamera frame'ini alır, bagaj tespitlerini döner.
    Check-in verisi, doluluk hesabı, dashboard → başka servislerin işi.

    PATTERN: Pipeline
    process_frame() şu adımları sırayla çalıştırır:
    1. detect()     → ham bbox'lar
    2. track()      → bbox'lara kalıcı ID ata
    3. classify()   → boyut sınıfı belirle
    4. filter()     → yeni tespit edilenleri işaretle

    PATTERN: Dependency Injection
    Detector ve Tracker dışarıdan verilir (inject edilir).
    InferenceService hangisini kullandığını bilmez — sadece arayüzü bilir.
    Test'te MockDetector, production'da YOLOv8Detector geçilir.
    """

    def __init__(
        self,
        gate_id: str,
        detector: BaseDetector,
        tracker:  BaseTracker,
        calibration: CalibrationResult,
        adaptive_threshold: Optional[AdaptiveThreshold] = None,
    ):
        self._gate_id    = gate_id
        self._detector   = detector
        self._tracker    = tracker
        self._calibration = calibration
        self._threshold  = adaptive_threshold or AdaptiveThreshold(
            base_threshold=0.45
        )

        # Gate state — private, dışarıdan değiştirilemez
        self.__sayilan_idler: dict[int, BoyutSinifi] = {}
        self.__frame_count = 0

        # Performans metrikleri
        self.__perf_window = deque(maxlen=50)

    def baslat(self) -> None:
        """Servisi başlat — detector warmup."""
        if not self._detector.is_ready:
            self._detector.warmup()
        logger.info(
            f"[{self._gate_id}] InferenceService hazir. "
            f"Kalibrasyon: {self._calibration.version}"
        )

    def process_frame(self, frame: np.ndarray) -> FrameSonucu:
        """
        Ana pipeline metodu.

        Tek sorumluluk: frame'i al, FrameSonucu döndür.
        Her adım ayrı private metod — bağımsız test edilebilir.
        """
        t_start = time.perf_counter()
        self.__frame_count += 1

        # Adım 1: Ham tespit
        raw_detections = self._detector.detect(
            frame, self._threshold.value
        )

        # Adaptive threshold güncelle
        confidences = [d["confidence"] for d in raw_detections]
        if confidences:
            self._threshold.update(confidences)

        # Adım 2: Tracking
        tracked = self._tracker.update(raw_detections)

        # Adım 3 + 4: Sınıflandır ve filtrele
        tespitler = []
        yeni_sayilanlar = []

        t_inf = time.perf_counter()

        for det in tracked:
            try:
                bbox = BoundingBox(*det["bbox"])
            except (ValueError, TypeError):
                continue

            bagaj_tipi = BagajTipi.from_coco_id(det.get("class_id", -1))

            # Boyut sınıfı — kalibrasyona delegate edilir
            boyut = self._classify_with_calibration(bbox)

            # İlk kez görüldü mü?
            tid = det["track_id"]
            is_new = tid not in self.__sayilan_idler

            if is_new:
                # İlk tespitte sınıfı kilitle
                self.__sayilan_idler[tid] = boyut

            # Kilitli sınıfı kullan (flickering önleme)
            locked_boyut = self.__sayilan_idler[tid]

            tespit = BagajTespiti(
                track_id=tid,
                bagaj_tipi=bagaj_tipi,
                boyut_sinifi=locked_boyut,
                bbox=bbox,
                confidence=det["confidence"],
                frame_no=self.__frame_count,
                gate_id=self._gate_id,
                sayildi_mi=is_new,
                metadata=BagajTespiti.Metadata(
                    kalibrasyon_version=self._calibration.version,
                    model_version=MODEL_VERSION,
                    inference_ms=round((t_inf - t_start) * 1000, 2),
                    timestamp_ms=int(time.time() * 1000),
                )
            )

            tespitler.append(tespit)
            if is_new:
                yeni_sayilanlar.append(tespit)

        t_end = time.perf_counter()
        isleme_ms = round((t_end - t_start) * 1000, 2)
        self.__perf_window.append(isleme_ms)

        return FrameSonucu(
            frame_no=self.__frame_count,
            gate_id=self._gate_id,
            tespitler=tespitler,
            isleme_ms=isleme_ms,
            yeni_sayilanlar=yeni_sayilanlar,
        )

    def _classify_with_calibration(self, bbox: BoundingBox) -> BoyutSinifi:
        """
        Kalibrasyon eşiklerine göre boyut sınıfı döner.

        NEDEN private?
        Bu metod InferenceService'in iç detayı.
        CalibrationResult.classify_bbox'ı kullanır — hesabı tekrarlamaz.
        """
        try:
            sinif_str = self._calibration.classify_bbox(bbox.to_tuple())
            return BoyutSinifi(sinif_str)
        except (RuntimeError, ValueError):
            return BoyutSinifi.UNKNOWN

    def update_calibration(self, new_calibration: CalibrationResult) -> None:
        """
        Kalibrasyon güncelleme — config server'dan yeni versiyon gelince.

        Mevcut sayım korunur — sadece gelecek tespitler yeni kalibrasyonla.
        """
        old_version = self._calibration.version
        self._calibration = new_calibration
        logger.info(
            f"[{self._gate_id}] Kalibrasyon guncellendi: "
            f"{old_version} → {new_calibration.version}"
        )

    def yeni_ucus_baslat(self) -> None:
        """
        Uçuş değişince sayaçları sıfırla.
        Tracker da sıfırlanır — eski ID'ler yeni uçuşa taşınmaz.
        """
        self.__sayilan_idler.clear()
        self.__frame_count = 0
        self._tracker.reset()
        logger.info(f"[{self._gate_id}] Yeni ucus basladi, sayaclar sifirlandi.")

    # ── Readonly Properties ──
    @property
    def toplam_sayilan(self) -> int:
        """
        Gate'den geçen TÜM bagaj nesnelerinin sayısı (personal item dahil).
        Genel istatistik/bilgi amaçlı — overhead bin doluluğu hesabı için
        KULLANILMAMALI, onun için bkz. overhead_bin_sayilan.
        """
        return len(self.__sayilan_idler)

    @property
    def overhead_bin_sayilan(self) -> int:
        """
        GEMINI 3. TUR, MADDE 3 (Seçenek B — kök neden düzeltmesi):
        Kabin üstü dolap (overhead bin) kapasitesini SADECE cabin_ok ve
        oversized bagajlar tüketir. THY kuralına göre personal item
        (örnek: sırt çantası, el çantası, laptop çantası) koltuk ALTINA
        gider, overhead bin'i hiç kullanmaz. Bu yüzden overhead bin
        doluluk hesabı yapılırken toplam_sayilan DEĞİL, bu property
        kullanılmalıdır — personal item bilinçli olarak hariç tutulur.

        Önceki davranış (düzeltme öncesi): toplam_sayilan personal item'ı
        da içeriyordu ve doğrudan dolap_kapasitesi ile karşılaştırılıyordu
        — bu, THY kuralına aykırı bir fazla-sayım hatasıydı.
        """
        dagilim = self.boyut_dagilimi
        return dagilim.get("cabin_ok", 0) + dagilim.get("oversized", 0)

    @property
    def boyut_dagilimi(self) -> dict[str, int]:
        sayac: dict[str, int] = {
            "oversized":     0,
            "cabin_ok":      0,
            "personal_item": 0,
            "unknown":       0,
        }
        for boyut in self.__sayilan_idler.values():
            sayac[boyut.value] = sayac.get(boyut.value, 0) + 1
        return sayac

    @property
    def ortalama_isleme_ms(self) -> float:
        if not self.__perf_window:
            return 0.0
        return round(sum(self.__perf_window) / len(self.__perf_window), 2)

    @property
    def gate_id(self) -> str:
        return self._gate_id

    @property
    def aktif_threshold(self) -> float:
        return self._threshold.value

    @property
    def scene_quality_flag(self) -> str:
        """Sabit threshold'u değiştirmeden sahne kalitesini raporlar."""
        return getattr(self._threshold, "quality_flag", "UNKNOWN")
