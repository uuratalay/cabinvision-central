# tests/edge/test_calibration_service.py
#
# KAVRAM: Unit Test
# Her servis kendi başına test edilebilir olmalı.
# Bu testler kameraya, modele, network'e ihtiyaç duymaz.
# Sadece saf Python mantığını test eder.
#
# Testler aynı zamanda "canlı dokümantasyon" işlevi görür:
# CalibrationService nasıl kullanılır? Test dosyasına bak.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from edge.models.calibration_models import (
    GatePhysicalParams,
    ReferenceObject,
    CalibrationResult,
)
from edge.services.calibration_service import (
    CalibrationService,
    ManualCalibrationStrategy,
    AutoCalibrationStrategy,
)


def test_gate_params_validation():
    """GatePhysicalParams validation kuralları çalışıyor mu?"""
    # Geçerli params
    params = GatePhysicalParams(
        gate_id="IST-GATE-12",
        camera_height_m=3.5,
        camera_tilt_deg=20.0,
        camera_fov_horizontal_deg=90.0,
        camera_fov_vertical_deg=60.0,
        frame_width_px=1280,
        frame_height_px=720,
    )
    assert params.gate_id == "IST-GATE-12"

    # Geçersiz yükseklik
    try:
        GatePhysicalParams(
            gate_id="TEST", camera_height_m=-1.0,
            camera_tilt_deg=0, camera_fov_horizontal_deg=90,
            camera_fov_vertical_deg=60,
            frame_width_px=1280, frame_height_px=720
        )
        assert False, "ValueError beklendi"
    except ValueError as e:
        assert "pozitif" in str(e).lower() or "height" in str(e).lower()

    print("  ✓ GatePhysicalParams validation")


def test_reference_object_properties():
    """ReferenceObject computed property'leri doğru mu?"""
    ref = ReferenceObject(
        real_width_cm=55.0,
        real_height_cm=40.0,
        pixel_bbox=(100, 200, 300, 400),  # 200x200 piksel
        distance_from_camera_m=2.0,
    )
    assert ref.pixel_width == 200
    assert ref.pixel_height == 200
    assert ref.pixel_area == 40000
    assert ref.real_area_cm2 == 2200.0
    print("  ✓ ReferenceObject properties")


def test_manual_calibration():
    """Manuel kalibrasyon doğru çalışıyor mu?"""
    service = CalibrationService.create("manual")

    params = GatePhysicalParams(
        gate_id="IST-GATE-12",
        camera_height_m=3.5,
        camera_tilt_deg=15.0,
        camera_fov_horizontal_deg=90.0,
        camera_fov_vertical_deg=60.0,
        frame_width_px=1280,
        frame_height_px=720,
    )

    # Standart kabin bagajı referans olarak kullanılıyor
    reference = ReferenceObject(
        real_width_cm=55.0,
        real_height_cm=40.0,
        pixel_bbox=(400, 300, 620, 520),  # 220x220 piksel
        distance_from_camera_m=2.0,
    )

    result = service.calibrate_gate(params, reference)

    assert result.is_valid, f"Kalibrasyon gecersiz: {result.notes}"
    assert result.gate_id == "IST-GATE-12"
    assert result.calibration_method == "manual"
    assert result.cm_per_pixel_at_ref_distance > 0
    assert "personal_item_max" in result.size_thresholds_px
    assert "cabin_ok_max" in result.size_thresholds_px
    assert result.size_thresholds_px["personal_item_max"] < result.size_thresholds_px["cabin_ok_max"]

    print(f"  ✓ Manuel kalibrasyon: cm/px={result.cm_per_pixel_at_ref_distance:.4f}")
    print(f"    Esikler: personal={result.size_thresholds_px['personal_item_max']:.0f}px, "
          f"cabin_ok={result.size_thresholds_px['cabin_ok_max']:.0f}px")


def test_auto_calibration():
    """Otomatik kalibrasyon doğru çalışıyor mu?"""
    service = CalibrationService.create("auto")

    params = GatePhysicalParams(
        gate_id="IST-GATE-07",
        camera_height_m=3.0,
        camera_tilt_deg=10.0,
        camera_fov_horizontal_deg=85.0,
        camera_fov_vertical_deg=55.0,
        frame_width_px=1280,
        frame_height_px=720,
    )

    result = service.calibrate_gate(params)

    assert result.is_valid
    assert result.calibration_method == "auto"
    assert result.cm_per_pixel_at_ref_distance > 0
    assert result.confidence_score > 0

    print(f"  ✓ Otomatik kalibrasyon: cm/px={result.cm_per_pixel_at_ref_distance:.4f}, "
          f"guven={result.confidence_score:.2f}")


def test_bbox_classification():
    """Boyut sınıflandırma doğru çalışıyor mu?"""
    service = CalibrationService.create("manual")
    params = GatePhysicalParams(
        gate_id="TEST-GATE",
        camera_height_m=3.5,
        camera_tilt_deg=0.0,
        camera_fov_horizontal_deg=90.0,
        camera_fov_vertical_deg=60.0,
        frame_width_px=1280,
        frame_height_px=720,
    )
    reference = ReferenceObject(
        real_width_cm=55.0, real_height_cm=40.0,
        pixel_bbox=(0, 0, 200, 180),  # ~0.27 cm/px
        distance_from_camera_m=3.5,
    )
    result = service.calibrate_gate(params, reference)

    thresholds = result.size_thresholds_px
    cabin_ok_max = thresholds["cabin_ok_max"]
    personal_max = thresholds["personal_item_max"]

    # Küçük bbox → personal item
    small_bbox = (0, 0, int(personal_max**0.5 * 0.7), int(personal_max**0.5 * 0.7))
    # Orta bbox → cabin ok
    mid_side = int(((personal_max + cabin_ok_max) / 2) ** 0.5)
    mid_bbox = (0, 0, mid_side, mid_side)
    # Büyük bbox → oversized
    large_side = int((cabin_ok_max * 1.5) ** 0.5)
    large_bbox = (0, 0, large_side, large_side)

    assert result.classify_bbox(small_bbox) == "personal_item"
    assert result.classify_bbox(mid_bbox) == "cabin_ok"
    assert result.classify_bbox(large_bbox) == "oversized"

    print(f"  ✓ Boyut siniflandirma:")
    print(f"    personal ({small_bbox[2]}x{small_bbox[3]}px) → personal_item")
    print(f"    cabin ({mid_bbox[2]}x{mid_bbox[3]}px) → cabin_ok")
    print(f"    oversize ({large_bbox[2]}x{large_bbox[3]}px) → oversized")


def test_cache_mechanism():
    """Cache mekanizması çalışıyor mu?"""
    service = CalibrationService.create("auto")
    params = GatePhysicalParams(
        gate_id="CACHE-TEST",
        camera_height_m=3.0, camera_tilt_deg=0.0,
        camera_fov_horizontal_deg=90.0, camera_fov_vertical_deg=60.0,
        frame_width_px=1280, frame_height_px=720,
    )

    result1 = service.calibrate_gate(params)
    result2 = service.calibrate_gate(params)  # cache'den gelmeli

    assert result1.version == result2.version  # aynı versiyon = cache'den geldi
    assert "CACHE-TEST" in service.calibrated_gates

    # Invalidate
    service.invalidate("CACHE-TEST")
    assert "CACHE-TEST" not in service.calibrated_gates

    # force_recalibrate
    result3 = service.calibrate_gate(params, force_recalibrate=True)
    assert result3.version != result1.version  # yeni versiyon

    print("  ✓ Cache mekanizmasi")


def test_strategy_switch():
    """Runtime strateji değişimi çalışıyor mu?"""
    service = CalibrationService.create("auto")
    assert service.active_strategy == "auto"

    service.switch_strategy("manual")
    assert service.active_strategy == "manual"

    # Manuel stratejiye geçince cache temizlendi mi?
    assert len(service.calibrated_gates) == 0

    print("  ✓ Runtime strateji degisimi")


def test_manual_without_reference_raises():
    """Manuel kalibrasyon referanssız hata vermeli."""
    service = CalibrationService.create("manual")
    params = GatePhysicalParams(
        gate_id="ERR-TEST",
        camera_height_m=3.0, camera_tilt_deg=0.0,
        camera_fov_horizontal_deg=90.0, camera_fov_vertical_deg=60.0,
        frame_width_px=1280, frame_height_px=720,
    )

    try:
        service.calibrate_gate(params, reference=None)
        assert False, "ValueError beklendi"
    except ValueError as e:
        assert "referans" in str(e).lower() or "reference" in str(e).lower()

    print("  ✓ Manuel kalibrasyonda referans zorunlulugu")


def test_confidence_scores():
    """Güven skorları mantıklı aralıkta mı?"""
    service_auto = CalibrationService.create("auto")
    params = GatePhysicalParams(
        gate_id="CONF-TEST",
        camera_height_m=3.5, camera_tilt_deg=20.0,
        camera_fov_horizontal_deg=90.0, camera_fov_vertical_deg=60.0,
        frame_width_px=1280, frame_height_px=720,
    )

    result_auto = service_auto.calibrate_gate(params)
    assert 0.0 <= result_auto.confidence_score <= 1.0

    service_manual = CalibrationService.create("manual")
    ref = ReferenceObject(
        real_width_cm=55.0, real_height_cm=40.0,
        pixel_bbox=(400, 300, 655, 520),
        distance_from_camera_m=2.0,
    )
    result_manual = service_manual.calibrate_gate(params, ref)
    assert 0.0 <= result_manual.confidence_score <= 1.0

    print(f"  ✓ Guven skorlari: auto={result_auto.confidence_score:.2f}, "
          f"manual={result_manual.confidence_score:.2f}")


if __name__ == "__main__":
    print("\n=== CalibrationService Testleri ===\n")
    tests = [
        test_gate_params_validation,
        test_reference_object_properties,
        test_manual_calibration,
        test_auto_calibration,
        test_bbox_classification,
        test_cache_mechanism,
        test_strategy_switch,
        test_manual_without_reference_raises,
        test_confidence_scores,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Sonuc: {passed} gecti, {failed} basarisiz")
    print(f"{'='*40}\n")
