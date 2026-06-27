# edge/services/calibration_service.py
#
# MİMARİ KARAR: Strategy Pattern
#
# Kalibrasyon iki şekilde yapılabilir: manuel (referans nesne tutarak)
# veya otomatik (kamera parametrelerinden hesaplayarak). Bu iki strateji
# farklı algoritmalar kullanır ama aynı çıktıyı üretir: CalibrationResult.
#
# Strategy Pattern burada idealdir çünkü:
# 1. Algoritma runtime'da seçilebilir (config'den gelir)
# 2. Yeni bir strateji eklemek mevcut kodu değiştirmez (Open/Closed Principle)
# 3. Her strateji bağımsız test edilebilir
#
# KAVRAM: Abstract Base Class (ABC)
# Python'da interface benzeri yapı ABC ile kurulur.
# @abstractmethod ile işaretlenen metodlar alt sınıflar tarafından
# MUTLAKA implement edilmek zorundadır. Edilmezse instantiation'da hata alınır.
# Bu compile-time güvencesi değil runtime güvencesidir — ama yine de etkilidir.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import math
import logging
import numpy as np

from edge.models.calibration_models import (
    GatePhysicalParams,
    ReferenceObject,
    CalibrationResult,
)

logger = logging.getLogger(__name__)

# THY standart kabin bagaj boyutları (cm) — kalibrasyon referansı
# Bu değerler THY bagaj kurallarından alınmıştır
STANDARD_CABIN_WIDTH_CM = 55.0
STANDARD_CABIN_HEIGHT_CM = 40.0
STANDARD_CABIN_DEPTH_CM = 25.0

# Boyut sınıfı sınırları (gerçek dünya, cm²)
#
# KAYNAK: turkishairlines.com resmi kabin bagajı sayfası (doğrulanmış, 2026)
#   Kabin bagajı (Economy):  55 x 40 x 23 cm, max 8 kg
#   Personal item:           40 x 30 x 15 cm, max 4 kg
#
# YÖNTEM NOTU (önemli sınırlama):
# Kamera tek RGB görüntü aldığından bagajın 3 boyutunu değil,
# yalnızca kameraya dönük YÜZEYİNİ (2D projeksiyon) görür. Burada kullanılan
# "ön yüzey alanı" (genişlik x yükseklik) bu yüzden KESİN bir ölçüm değil,
# zayıf bir vekil (proxy) göstergedir. Bu sınırlama nedeniyle sistem net bir
# "cm cinsinden ölçtüm" iddiası taşımaz — bkz. BoyutSinifi.UNCERTAIN bandı
# ve classify_bbox() içindeki belirsizlik marjı.
#
# personal_item: kameraya dönük yüzey ~ 40x30 cm  -> 1200 cm²
# cabin_ok:      kameraya dönük yüzey ~ 55x40 cm  -> 2200 cm²
# cabin_ok_max üzeri → oversized (yine belirsizlik marjıyla)
PERSONAL_ITEM_MAX_AREA_CM2 = 1200.0    # 40x30 cm (THY personal item resmi ölçüsü)
CABIN_OK_MAX_AREA_CM2 = 2200.0         # 55x40 cm (THY kabin bagajı resmi ölçüsü)

# Belirsizlik marjı: eşiklere yakın (±%15) bbox'lar kesin sınıf yerine
# UNCERTAIN olarak işaretlenir. Bu, "sınırda" tespitlerde sahte kesinlik
# üretmemek için var.
SINIRDA_BELIRSIZLIK_ORANI = 0.15


class BaseCalibrationStrategy(ABC):
    """
    Tüm kalibrasyon stratejilerinin soyut temel sınıfı.

    KAVRAM: Template Method Pattern (gömülü)
    calibrate() metodu alt sınıflar tarafından implement edilir.
    validate() metodu burada concrete olarak tanımlanır — tüm stratejiler
    aynı validation kurallarını kullanır. Bu Template Method'un basit hali.
    """

    @abstractmethod
    def calibrate(
        self,
        params: GatePhysicalParams,
        reference: Optional[ReferenceObject] = None,
    ) -> CalibrationResult:
        """
        Kalibrasyon yapar ve sonuç döner.

        Alt sınıflar bu metodu override ETMEK ZORUNDADIR.
        Override edilmezse TypeError fırlatılır.

        Args:
            params: Gate'in fiziksel parametreleri
            reference: Manuel kalibrasyon için referans nesne (opsiyonel)

        Returns:
            CalibrationResult: Hesaplanan kalibrasyon sonucu
        """
        ...

    def validate(self, result: CalibrationResult) -> bool:
        """
        Kalibrasyon sonucunu doğrular.

        NEDEN concrete (abstract değil)?
        Validation kuralları tüm stratejiler için aynı.
        Alt sınıfların bunu override etmesine gerek yok.
        Ama gerekirse override EDEBİLİRLER (kısıtlama yok).
        """
        if result.cm_per_pixel_at_ref_distance <= 0:
            logger.warning(
                f"[{result.gate_id}] Gecersiz cm/pixel orani: "
                f"{result.cm_per_pixel_at_ref_distance}"
            )
            return False

        if not result.size_thresholds_px:
            logger.warning(f"[{result.gate_id}] Esik degerleri bos")
            return False

        # Gerçekçi aralık kontrolü: 1280x720 frame'de
        # çok küçük veya çok büyük değerler kalibrasyon hatasına işaret eder
        cabin_ok_threshold = result.size_thresholds_px.get("cabin_ok_max", 0)
        if not (500 < cabin_ok_threshold < 200000):
            logger.warning(
                f"[{result.gate_id}] Cabin OK esigi gercekci degil: "
                f"{cabin_ok_threshold}"
            )
            return False

        return True

    def _generate_version(self, gate_id: str, method: str) -> str:
        """Kalibrasyon versiyonu üretir. Format: gate_id-method-timestamp-random"""
        import time, random
        ts = int(time.time())
        uid = random.randint(1000, 9999)
        return f"{gate_id}-{method}-{ts}-{uid}"

    def _cm2_to_pixel_area(
        self,
        area_cm2: float,
        cm_per_pixel: float,
        distance_factor: float = 1.0,
    ) -> float:
        """
        Gerçek dünya cm² → piksel alanı dönüşümü.

        NEDEN distance_factor?
        Kameraya yakın nesneler aynı cm² için daha fazla piksel kaplar.
        distance_factor bu perspektif etkisini düzeltir.
        """
        pixel_per_cm = 1.0 / cm_per_pixel
        return area_cm2 * (pixel_per_cm ** 2) * distance_factor


class ManualCalibrationStrategy(BaseCalibrationStrategy):
    """
    Manuel kalibrasyon stratejisi.

    NASIL ÇALIŞIR:
    Gate'e bilinen boyutlarda bir referans nesne tutulur.
    Kameradaki piksel boyutu ile gerçek cm boyutu karşılaştırılır.
    Bu oran tüm boyut hesaplamaları için kullanılır.

    AVANTAJ: Basit, doğruluğu yüksek, o gate'e özgü
    DEZAVANTAJ: Her gate için manuel işlem gerektirir
    """

    def calibrate(
        self,
        params: GatePhysicalParams,
        reference: Optional[ReferenceObject] = None,
    ) -> CalibrationResult:
        """
        BaseCalibrationStrategy.calibrate() override.

        Args:
            params: Gate fiziksel parametreleri
            reference: ZORUNLU — referans nesne olmadan çalışmaz

        Raises:
            ValueError: Referans nesne sağlanmamışsa
        """
        if reference is None:
            raise ValueError(
                "Manuel kalibrasyon icin referans nesne zorunludur. "
                "Gate'e bilinen boyutlarda bir bagaj tutun ve "
                "ReferenceObject olusturun."
            )

        logger.info(
            f"[{params.gate_id}] Manuel kalibrasyon basliyor. "
            f"Referans: {reference.real_width_cm}x{reference.real_height_cm}cm "
            f"@ {reference.pixel_width}x{reference.pixel_height}px"
        )

        # cm/piksel oranını hesapla — yatay ve dikey ortalaması
        cm_per_pixel_h = reference.real_width_cm / max(reference.pixel_width, 1)
        cm_per_pixel_v = reference.real_height_cm / max(reference.pixel_height, 1)
        cm_per_pixel = (cm_per_pixel_h + cm_per_pixel_v) / 2.0

        logger.debug(
            f"[{params.gate_id}] cm/pixel: yatay={cm_per_pixel_h:.4f}, "
            f"dikey={cm_per_pixel_v:.4f}, ortalama={cm_per_pixel:.4f}"
        )

        # Mesafe faktörü: referans nesne ne kadar uzaktaydı
        distance_factor = self._calculate_distance_factor(
            reference.distance_from_camera_m,
            params.camera_height_m,
        )

        # Piksel eşiklerini hesapla
        thresholds = self._calculate_thresholds(cm_per_pixel, distance_factor)

        version = self._generate_version(params.gate_id, "manual")

        result = CalibrationResult(
            gate_id=params.gate_id,
            version=version,
            cm_per_pixel_at_ref_distance=cm_per_pixel,
            size_thresholds_px=thresholds,
            homography_matrix=None,  # manuel kalibrasyonda homografi yok
            calibration_method="manual",
            confidence_score=self._calculate_confidence(
                cm_per_pixel_h, cm_per_pixel_v
            ),
            notes=(
                f"Referans: {reference.real_width_cm}x{reference.real_height_cm}cm, "
                f"mesafe: {reference.distance_from_camera_m}m"
            ),
        )

        result.is_valid = self.validate(result)

        if result.is_valid:
            logger.info(
                f"[{params.gate_id}] Manuel kalibrasyon basarili. "
                f"Versiyon: {version}, Guven: {result.confidence_score:.2f}"
            )
        else:
            logger.error(
                f"[{params.gate_id}] Manuel kalibrasyon basarisiz. "
                "Parametreleri kontrol edin."
            )

        return result

    def _calculate_distance_factor(
        self,
        distance_m: float,
        camera_height_m: float,
    ) -> float:
        """
        Mesafe faktörü: nesnenin referans mesafesine göre
        perspektif büyüme/küçülme faktörü.

        Pinhole kamera modeli: boyut ∝ 1/mesafe
        Referans mesafe = camera_height (kameranın tam altı)
        """
        if distance_m <= 0:
            return 1.0
        return camera_height_m / distance_m

    def _calculate_thresholds(
        self,
        cm_per_pixel: float,
        distance_factor: float,
    ) -> dict[str, float]:
        """
        Gerçek dünya cm² değerlerini piksel alanlarına dönüştürür.
        """
        personal_px = self._cm2_to_pixel_area(
            PERSONAL_ITEM_MAX_AREA_CM2, cm_per_pixel, distance_factor
        )
        cabin_ok_px = self._cm2_to_pixel_area(
            CABIN_OK_MAX_AREA_CM2, cm_per_pixel, distance_factor
        )

        return {
            "personal_item_max": round(personal_px),
            "cabin_ok_max": round(cabin_ok_px),
        }

    def _calculate_confidence(
        self,
        cm_per_pixel_h: float,
        cm_per_pixel_v: float,
    ) -> float:
        """
        Yatay ve dikey cm/pixel değerlerinin tutarlılığından güven skoru.
        İkisi çok farklıysa referans nesne eğik tutulmuş olabilir.
        """
        if cm_per_pixel_h <= 0 or cm_per_pixel_v <= 0:
            return 0.0
        ratio = min(cm_per_pixel_h, cm_per_pixel_v) / max(
            cm_per_pixel_h, cm_per_pixel_v
        )
        # ratio=1.0 → mükemmel, ratio<0.8 → şüpheli
        return round(min(ratio, 1.0), 3)


class AutoCalibrationStrategy(BaseCalibrationStrategy):
    """
    Otomatik kalibrasyon stratejisi.

    NASIL ÇALIŞIR:
    Kamera yüksekliği, açısı ve görüş açısı (FOV) parametrelerinden
    perspektif geometri formülleri ile cm/pixel oranı hesaplanır.
    Ek olarak perspektif distorsiyon düzeltmesi için homografi matrisi üretilir.

    AVANTAJ:
    - Manuel işlem gerektirmez
    - Gate parametreleri değişince otomatik recalibration mümkün
    - Homografi ile perspektif normalizasyonu yapılır (özgünlük)

    DEZAVANTAJ:
    - Kamera parametrelerinin doğru girilmesi kritik
    - Manuel kalibrasyona göre daha düşük doğruluk (parametreler yaklaşık)
    """

    def calibrate(
        self,
        params: GatePhysicalParams,
        reference: Optional[ReferenceObject] = None,
    ) -> CalibrationResult:
        """
        BaseCalibrationStrategy.calibrate() override.

        Referans nesne olmadan çalışır — sadece fiziksel parametreler yeterli.
        Referans nesne verilirse confidence skoru artırılır (hibrit mod).
        """
        logger.info(
            f"[{params.gate_id}] Otomatik kalibrasyon basliyor. "
            f"Yukseklik: {params.camera_height_m}m, "
            f"Aci: {params.camera_tilt_deg}deg"
        )

        # Temel cm/pixel hesabı — trigonometri
        cm_per_pixel = self._calculate_cm_per_pixel(params)

        # Perspektif homografi matrisi
        homography = self._compute_homography(params)

        # Eşikler
        thresholds = self._calculate_thresholds_from_geometry(
            params, cm_per_pixel
        )

        # Referans nesne varsa confidence boost
        confidence = self._base_confidence(params)
        if reference is not None:
            confidence = self._boost_confidence_with_reference(
                params, reference, cm_per_pixel, confidence
            )

        version = self._generate_version(params.gate_id, "auto")

        result = CalibrationResult(
            gate_id=params.gate_id,
            version=version,
            cm_per_pixel_at_ref_distance=cm_per_pixel,
            size_thresholds_px=thresholds,
            homography_matrix=homography,
            calibration_method="auto",
            confidence_score=confidence,
            notes=(
                f"Geometri bazli: yukseklik={params.camera_height_m}m, "
                f"aci={params.camera_tilt_deg}deg, "
                f"fov={params.camera_fov_horizontal_deg}deg"
            ),
        )

        result.is_valid = self.validate(result)

        if result.is_valid:
            logger.info(
                f"[{params.gate_id}] Otomatik kalibrasyon basarili. "
                f"Versiyon: {version}, Guven: {confidence:.2f}"
            )

        return result

    def _calculate_cm_per_pixel(self, params: GatePhysicalParams) -> float:
        """
        Kamera geometrisinden cm/pixel hesabı.

        Formül:
        Kamera merkezinin altındaki nokta için (tilt=0, nadir view):
          ground_width = 2 * height * tan(fov_h / 2)
          cm_per_pixel = ground_width_cm / frame_width_px

        Tilt varsa projeksiyon düzeltmesi eklenir.
        """
        fov_h_rad = math.radians(params.camera_fov_horizontal_deg)
        tilt_rad = math.radians(params.camera_tilt_deg)

        # Kameranın yere dik projeksiyon mesafesi
        effective_height = params.camera_height_m * math.cos(tilt_rad)

        # Görüş alanının yerdeki genişliği (metre)
        ground_width_m = 2 * effective_height * math.tan(fov_h_rad / 2)

        # m → cm
        ground_width_cm = ground_width_m * 100

        # cm / piksel
        cm_per_pixel = ground_width_cm / params.frame_width_px

        logger.debug(
            f"[{params.gate_id}] Geometri: "
            f"effective_height={effective_height:.2f}m, "
            f"ground_width={ground_width_cm:.1f}cm, "
            f"cm_per_pixel={cm_per_pixel:.4f}"
        )

        return cm_per_pixel

    def _compute_homography(
        self,
        params: GatePhysicalParams,
    ) -> Optional[np.ndarray]:
        """
        Perspektif düzeltme için homografi matrisi hesaplar.

        Kamera aşağıdan yukarı veya açılı baktığında zemin nesneleri
        perspektif distorsiyona uğrar. Homografi bu distorsiyonu düzeltir.

        4 nokta yöntemi:
        - Frame'in 4 köşesinin ground plane'deki karşılığını hesapla
        - Bu 4 nokta çiftiyle homografi matrisi üret
        - Bu matris ile her bbox normalize edilebilir
        """
        if params.camera_tilt_deg < 5.0:
            # Neredeyse dik bakış — homografi gerekmez
            logger.debug(
                f"[{params.gate_id}] Tilt <5 derece, "
                "homografi hesaplanmadi"
            )
            return None

        tilt_rad = math.radians(params.camera_tilt_deg)
        fov_h_rad = math.radians(params.camera_fov_horizontal_deg)
        fov_v_rad = math.radians(params.camera_fov_vertical_deg)
        h = params.camera_height_m
        W = params.frame_width_px
        H = params.frame_height_px

        # Frame köşelerinin ground plane'deki koordinatları (metre)
        # Basitleştirilmiş pinhole model
        src_pts = np.float32([
            [0, 0], [W, 0], [W, H], [0, H]
        ])

        # Ground plane koordinatları (perspektif projeksiyon)
        half_w_angle = fov_h_rad / 2
        half_v_angle = fov_v_rad / 2

        def ground_x(px_x: float) -> float:
            angle = (px_x / W - 0.5) * fov_h_rad
            return h * math.tan(angle) * 100  # cm

        def ground_y(px_y: float) -> float:
            angle = tilt_rad + (0.5 - px_y / H) * fov_v_rad
            if math.cos(angle) < 1e-6:
                return h * 100
            return h / math.tan(angle) * 100  # cm

        # Normalize edilmiş hedef frame (ortogonal görünüm)
        scale = W / (2 * h * math.tan(half_w_angle) * 100)
        dst_pts = np.float32([
            [ground_x(0) * scale + W/2, -ground_y(0) * scale + H/2],
            [ground_x(W) * scale + W/2, -ground_y(0) * scale + H/2],
            [ground_x(W) * scale + W/2, -ground_y(H) * scale + H/2],
            [ground_x(0) * scale + W/2, -ground_y(H) * scale + H/2],
        ])

        try:
            M, _ = cv2.findHomography(src_pts, dst_pts)
            logger.debug(f"[{params.gate_id}] Homografi matrisi hesaplandi")
            return M
        except Exception as e:
            logger.warning(
                f"[{params.gate_id}] Homografi hesaplanamadi: {e}"
            )
            return None

    def _calculate_thresholds_from_geometry(
        self,
        params: GatePhysicalParams,
        cm_per_pixel: float,
    ) -> dict[str, float]:
        """
        Kamera yüksekliğine göre perspektif-aware eşik hesabı.

        Önemli fark: Eşikler zemin seviyesindeki ortalama mesafe için
        hesaplanır, sadece kamera altı için değil.
        """
        # Ortalama boarding mesafesi: kameranın altından ~2m önde
        # Bu boarding gate gerçeğine yakın bir varsayım
        avg_distance_m = params.camera_height_m * 1.5
        distance_factor = params.camera_height_m / avg_distance_m

        personal_px = self._cm2_to_pixel_area(
            PERSONAL_ITEM_MAX_AREA_CM2, cm_per_pixel, distance_factor
        )
        cabin_ok_px = self._cm2_to_pixel_area(
            CABIN_OK_MAX_AREA_CM2, cm_per_pixel, distance_factor
        )

        return {
            "personal_item_max": round(personal_px),
            "cabin_ok_max": round(cabin_ok_px),
        }

    def _base_confidence(self, params: GatePhysicalParams) -> float:
        """
        Sadece geometri tabanlı kalibrasyonun güven skoru.
        Parametre kalitesine göre 0.5-0.75 aralığında.
        """
        score = 0.75

        # Yüksek tilt açısı → daha fazla distorsiyon → daha düşük güven
        if params.camera_tilt_deg > 45:
            score -= 0.15
        elif params.camera_tilt_deg > 30:
            score -= 0.08

        # Çok yüksek veya çok alçak kamera → güven düşer
        if params.camera_height_m < 2.0 or params.camera_height_m > 8.0:
            score -= 0.1

        return round(max(0.3, score), 3)

    def _boost_confidence_with_reference(
        self,
        params: GatePhysicalParams,
        reference: ReferenceObject,
        calculated_cm_per_pixel: float,
        base_confidence: float,
    ) -> float:
        """
        Referans nesne varsa hesaplanan değer ile karşılaştır,
        tutarlıysa confidence artır.

        Bu hibrit mod: auto hesapla, referansla doğrula.
        """
        ref_cm_per_pixel = (
            reference.real_width_cm / max(reference.pixel_width, 1)
        )
        error = abs(calculated_cm_per_pixel - ref_cm_per_pixel) / ref_cm_per_pixel

        if error < 0.05:   # %5'ten az hata → çok iyi
            boost = 0.15
        elif error < 0.15: # %15'ten az → kabul edilebilir
            boost = 0.08
        else:               # Büyük fark → parametreler şüpheli
            boost = -0.1
            logger.warning(
                f"[{params.gate_id}] Referans ile hesaplanan deger "
                f"arasinda %{error*100:.1f} fark var. "
                "Kamera parametrelerini kontrol edin."
            )

        return round(min(0.95, base_confidence + boost), 3)


# cv2 import'u burada — sadece homografi gerektiğinde
try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning(
        "OpenCV yuklu degil. Homografi hesaplama devre disi. "
        "pip install opencv-python-headless"
    )


class CalibrationService:
    """
    Kalibrasyon işlemlerini yöneten ana servis.

    KAVRAM: Context class (Strategy Pattern'in Context'i)
    Strategy Pattern'de Context, hangi stratejiyi kullanacağını bilir
    ama stratejinin içini bilmez. CalibrationService bu role sahip.

    KAVRAM: Factory Method (hafif)
    create() class method'u doğru stratejiyle CalibrationService döner.
    Client'ın strateji sınıflarını bilmesine gerek kalmaz.

    KAVRAM: Encapsulation
    _strategy ve _calibration_cache private. Dışarıdan erişilemez.
    Sadece public metodlar üzerinden etkileşim kurulur.
    """

    # Desteklenen metodlar — magic string'leri önlemek için
    METHOD_MANUAL = "manual"
    METHOD_AUTO = "auto"

    def __init__(self, strategy: BaseCalibrationStrategy):
        self._strategy = strategy
        # Cache: gate_id → CalibrationResult
        # Aynı gate için tekrar kalibrasyon yapılmasını önler
        self._calibration_cache: dict[str, CalibrationResult] = {}

    @classmethod
    def create(
        cls,
        method: str = METHOD_AUTO,
    ) -> "CalibrationService":
        """
        Factory method: doğru stratejiyle CalibrationService oluşturur.

        KAVRAM: Factory Method Pattern
        Client şunu yazar: CalibrationService.create("auto")
        Hangi strategy class'ının instantiate edileceğini bilmez.
        Yeni bir strateji eklendiğinde sadece bu metod değişir.

        Args:
            method: "manual" veya "auto"

        Returns:
            CalibrationService: İstenen stratejiyle yapılandırılmış servis
        """
        strategies = {
            cls.METHOD_MANUAL: ManualCalibrationStrategy,
            cls.METHOD_AUTO:   AutoCalibrationStrategy,
        }

        if method not in strategies:
            raise ValueError(
                f"Bilinmeyen kalibrasyon metodu: '{method}'. "
                f"Gecerli: {list(strategies.keys())}"
            )

        strategy_class = strategies[method]
        logger.info(f"CalibrationService olusturuldu: {method} stratejisi")
        return cls(strategy=strategy_class())

    def calibrate_gate(
        self,
        params: GatePhysicalParams,
        reference: Optional[ReferenceObject] = None,
        force_recalibrate: bool = False,
    ) -> CalibrationResult:
        """
        Gate'i kalibre eder.

        Cache mekanizması: Aynı gate daha önce kalibre edildiyse
        cache'den döner. force_recalibrate=True ile zorla yenilenir.

        Args:
            params: Gate fiziksel parametreleri
            reference: Referans nesne (manuel veya hibrit mod)
            force_recalibrate: True ise cache bypass edilir

        Returns:
            CalibrationResult
        """
        cache_key = params.gate_id

        if not force_recalibrate and cache_key in self._calibration_cache:
            cached = self._calibration_cache[cache_key]
            if cached.is_valid:
                logger.info(
                    f"[{params.gate_id}] Cache'den kalibrasyon donuluyor. "
                    f"Versiyon: {cached.version}"
                )
                return cached

        # Stratejiyi çalıştır
        result = self._strategy.calibrate(params, reference)

        # Cache'e kaydet (geçerli olsun ya da olmasın — geçersiz olanı da loglamak için)
        self._calibration_cache[cache_key] = result

        return result

    def get_calibration(self, gate_id: str) -> Optional[CalibrationResult]:
        """
        Cache'deki kalibrasyonu döner. Yoksa None.

        Bu metod sadece okuma yapar — yan etkisi yok (pure query).
        CQRS prensibinin hafif uygulaması: sorgular ve komutlar ayrı.
        """
        return self._calibration_cache.get(gate_id)

    def invalidate(self, gate_id: str) -> None:
        """
        Gate'in kalibrasyonunu geçersiz kılar.

        Kamera fiziksel olarak taşındığında veya yeniden konumlandırıldığında
        çağrılır. Sonraki calibrate_gate çağrısı yeniden hesaplar.
        """
        if gate_id in self._calibration_cache:
            self._calibration_cache[gate_id].is_valid = False
            logger.info(
                f"[{gate_id}] Kalibrasyon gecersiz kilindi. "
                "Yeniden kalibrasyon gerekiyor."
            )

    def switch_strategy(self, method: str) -> None:
        """
        Stratejiyi runtime'da değiştirir.

        KAVRAM: Strategy Pattern'in gücü burada.
        Sistemi yeniden başlatmadan manual→auto veya auto→manual geçiş.
        Yeni strateji sonraki calibrate_gate çağrısında devreye girer.
        Cache temizlenir — eski stratejiyle yapılan kalibrasyonlar geçersiz.
        """
        strategies = {
            self.METHOD_MANUAL: ManualCalibrationStrategy,
            self.METHOD_AUTO:   AutoCalibrationStrategy,
        }

        if method not in strategies:
            raise ValueError(f"Bilinmeyen strateji: {method}")

        old_strategy = type(self._strategy).__name__
        self._strategy = strategies[method]()
        self._calibration_cache.clear()

        logger.info(
            f"Strateji degistirildi: {old_strategy} → "
            f"{type(self._strategy).__name__}. Cache temizlendi."
        )

    @property
    def active_strategy(self) -> str:
        """Aktif stratejinin adını döner."""
        name = type(self._strategy).__name__
        return name.replace("CalibrationStrategy", "").lower()

    @property
    def calibrated_gates(self) -> list[str]:
        """Kalibre edilmiş gate'lerin listesi."""
        return [
            gid for gid, result in self._calibration_cache.items()
            if result.is_valid
        ]
