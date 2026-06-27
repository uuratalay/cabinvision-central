# tests/test_integration.py
#
# CabinVision uçtan uca entegrasyon testleri.
# Bu dosya artık standart pytest ile doğrudan çalışır.
# Önceki sürümde test fonksiyonları birbirine argüman geçirdiği için pytest
# bu argümanları fixture sanıyor ve `pytest -q` dört hata üretiyordu.

import os
import sys
import time
import logging
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.WARNING)

from edge.models.calibration_models import GatePhysicalParams, ReferenceObject
from edge.services.calibration_service import CalibrationService
from edge.services.inference_service import InferenceService, MockDetector, SimpleTracker, AdaptiveThreshold
from edge.services.event_publisher import EdgeEventPublisher

from central.models.flight_models import UcusBilgisi, UcakTipi, DolulukSeviyesi
from central.repositories.flight_memory_repository import FlightMemoryRepository, UcusKaydi
from central.services.prediction_engine import PredictionEngine
from central.services.fusion_engine import (
    FusionEngine,
    KuralBazliFusion,
    ActionService,
    GateDashboardObserver,
    CentralDashboardObserver,
    AuditLogObserver,
)


def bench(label: str, fn, *args, **kwargs):
    """Basit benchmark wrapper."""
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    ms = (time.perf_counter() - t) * 1000
    print(f"  ⏱  {label}: {ms:.2f}ms")
    return result, ms


@pytest.fixture
def calibration():
    """Gate kalibrasyonu: inference testlerinin ortak bağımlılığı."""
    params = GatePhysicalParams(
        gate_id="IST-GATE-12",
        camera_height_m=3.5,
        camera_tilt_deg=15.0,
        camera_fov_horizontal_deg=90.0,
        camera_fov_vertical_deg=60.0,
        frame_width_px=1280,
        frame_height_px=720,
    )
    ref = ReferenceObject(
        real_width_cm=55.0,
        real_height_cm=40.0,
        pixel_bbox=(400, 300, 620, 520),
        distance_from_camera_m=2.0,
    )
    service = CalibrationService.create("manual")
    result = service.calibrate_gate(params, ref)
    assert result.is_valid
    return result


@pytest.fixture
def inference_service(calibration):
    detector = MockDetector(seed=42, avg_detections=4)
    tracker = SimpleTracker(iou_threshold=0.25)
    # AdaptiveThreshold adı geriye dönük uyumluluk için korunuyor; yeni sürümde
    # eşik gizlice oynamaz, sahne kalitesini ayrı takip eder.
    threshold = AdaptiveThreshold(base_threshold=0.45)
    service = InferenceService(
        gate_id="IST-GATE-12",
        detector=detector,
        tracker=tracker,
        calibration=calibration,
        adaptive_threshold=threshold,
    )
    service.baslat()
    return service


@pytest.fixture
def memory_repo():
    repo = FlightMemoryRepository(window_size=20)
    rng = random.Random(99)
    hatlar = ["IST-DXB", "IST-LHR", "IST-AYT"]

    for i in range(30):
        hat = hatlar[i % 3]
        toplam = int(rng.uniform(60, 100))
        oversized = int(rng.uniform(5, 20))
        cabin_ok = int(toplam * 0.65)
        personal = max(0, toplam - oversized - cabin_ok)
        # Overhead hesabı personal item'ı dışarıda bırakır.
        doluluk = min((oversized + cabin_ok) / UcakTipi.NARROW_BODY.kapasite, 1.0)
        repo.kayit_ekle(UcusKaydi(
            ucus_no=f"TK-{1000+i}",
            hat=hat,
            toplam_bagaj=toplam,
            oversized_sayisi=oversized,
            cabin_ok_sayisi=cabin_ok,
            personal_sayisi=personal,
            doluluk_orani=doluluk,
            ucak_tipi="narrow_body",
        ))
    return repo


@pytest.fixture
def prediction_pair(memory_repo):
    engine = PredictionEngine.create("rule_based", memory=memory_repo)
    ucus_normal = UcusBilgisi(
        ucus_no="TK-2001",
        hat="IST-DXB",
        ucak_tipi=UcakTipi.NARROW_BODY,
        toplam_yolcu=168,
        cabin_beyan_sayisi=75,
        oversized_beyan=6,
        gate_id="IST-GATE-12",
    )
    ucus_riskli = UcusBilgisi(
        ucus_no="TK-2002",
        hat="IST-DXB",
        ucak_tipi=UcakTipi.NARROW_BODY,
        toplam_yolcu=180,
        cabin_beyan_sayisi=140,
        oversized_beyan=22,
        gate_id="IST-GATE-07",
    )
    return engine.tahmin_uret(ucus_riskli), engine.tahmin_uret(ucus_normal)


# ─────────────────────────────────────────────
# TEST 1: Kalibrasyon
# ─────────────────────────────────────────────
def test_kalibrasyon(calibration):
    print("\n[TEST 1] Kalibrasyon")
    assert calibration.confidence_score > 0.7
    assert "personal_item_max" in calibration.size_thresholds_px
    assert "cabin_ok_max" in calibration.size_thresholds_px
    print(f"  ✓ cm/px={calibration.cm_per_pixel_at_ref_distance:.4f}, "
          f"guven={calibration.confidence_score:.2f}")


# ─────────────────────────────────────────────
# TEST 2: InferenceService — frame işleme hızı
# ─────────────────────────────────────────────
def test_inference(inference_service):
    print("\n[TEST 2] InferenceService — Frame Isleme")
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    n_frames = 100

    t_start = time.perf_counter()
    for _ in range(n_frames):
        inference_service.process_frame(dummy_frame)
    total_ms = (time.perf_counter() - t_start) * 1000
    avg_ms = total_ms / n_frames

    print(f"  ✓ {n_frames} frame islendi")
    print(f"  ✓ Ortalama frame isleme: {avg_ms:.2f}ms ({1000/avg_ms:.1f} FPS kapasitesi)")
    print(f"  ✓ Toplam sayilan: {inference_service.toplam_sayilan} bagaj")
    print(f"  ✓ Aktif threshold: {inference_service.aktif_threshold}")

    assert avg_ms < 50
    assert inference_service.toplam_sayilan > 0


# ─────────────────────────────────────────────
# TEST 3: EventPublisher — outbox ve DLQ
# ─────────────────────────────────────────────
def test_event_publisher(inference_service):
    print("\n[TEST 3] EventPublisher — Outbox & DLQ")
    publisher = EdgeEventPublisher(
        gate_id="IST-GATE-12",
        batch_size=5,
        flush_interval_s=0.1,
    )
    publisher.baslat()

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    inference_service.yeni_ucus_baslat()
    for _ in range(50):
        frame_result = inference_service.process_frame(dummy_frame)
        publisher.publish_frame_result(frame_result)

    time.sleep(0.5)
    publisher.durdur()

    print(f"  ✓ Gonderilen: {publisher.sent_count}")
    print(f"  ✓ DLQ: {publisher.dlq_size}")
    assert publisher.sent_count > 0


# ─────────────────────────────────────────────
# TEST 4: FlightMemoryRepository
# ─────────────────────────────────────────────
def test_flight_memory(memory_repo):
    print("\n[TEST 4] FlightMemoryRepository — Sefer Hafizasi")
    for hat in ["IST-DXB", "IST-LHR", "IST-AYT"]:
        ist, _ = bench(f"Hat istatistigi ({hat})", memory_repo.hat_istatistigi_al, hat)
        assert ist is not None
        print(f"  ✓ {hat}: ort_doluluk={ist.ort_doluluk:.2f}, "
              f"kayit={ist.kayit_sayisi}, guven={ist.guven_skoru:.2f}")
    assert memory_repo.toplam_kayit_sayisi == 30


# ─────────────────────────────────────────────
# TEST 5: PredictionEngine
# ─────────────────────────────────────────────
def test_prediction_engine(memory_repo):
    print("\n[TEST 5] PredictionEngine")
    engine = PredictionEngine.create("rule_based", memory=memory_repo)
    ucus_normal = UcusBilgisi(
        ucus_no="TK-2001",
        hat="IST-DXB",
        ucak_tipi=UcakTipi.NARROW_BODY,
        toplam_yolcu=168,
        cabin_beyan_sayisi=75,
        oversized_beyan=6,
        gate_id="IST-GATE-12",
    )
    ucus_riskli = UcusBilgisi(
        ucus_no="TK-2002",
        hat="IST-DXB",
        ucak_tipi=UcakTipi.NARROW_BODY,
        toplam_yolcu=180,
        cabin_beyan_sayisi=140,
        oversized_beyan=22,
        gate_id="IST-GATE-07",
    )

    tahmin_normal, _ = bench("Normal ucus tahmini", engine.tahmin_uret, ucus_normal)
    tahmin_riskli, _ = bench("Riskli ucus tahmini", engine.tahmin_uret, ucus_riskli)
    _, cache_ms = bench("Cache'den tahmin", engine.tahmin_uret, ucus_normal)

    print(f"  ✓ Normal: doluluk=%{int(tahmin_normal.tahmini_doluluk_orani*100)}, "
          f"bagaj={tahmin_normal.tahmini_toplam_bagaj}, guven={tahmin_normal.guven_skoru:.2f}")
    print(f"  ✓ Riskli: doluluk=%{int(tahmin_riskli.tahmini_doluluk_orani*100)}, "
          f"bagaj={tahmin_riskli.tahmini_toplam_bagaj}, guven={tahmin_riskli.guven_skoru:.2f}")

    assert cache_ms < 1.0
    assert tahmin_riskli.tahmini_doluluk_orani > tahmin_normal.tahmini_doluluk_orani


# ─────────────────────────────────────────────
# TEST 6: FusionEngine + ActionService
# ─────────────────────────────────────────────
def test_fusion_ve_action(prediction_pair):
    print("\n[TEST 6] FusionEngine + ActionService")
    tahmin_riskli, _ = prediction_pair
    fusion = FusionEngine(KuralBazliFusion())
    action = ActionService()

    gate_obs = GateDashboardObserver("IST-GATE-07")
    central_obs = CentralDashboardObserver()
    audit_obs = AuditLogObserver()
    action.subscribe(gate_obs)
    action.subscribe(central_obs)
    action.subscribe(audit_obs)

    ucus = UcusBilgisi(
        ucus_no="TK-2002",
        hat="IST-DXB",
        ucak_tipi=UcakTipi.NARROW_BODY,
        toplam_yolcu=180,
        cabin_beyan_sayisi=140,
        oversized_beyan=22,
        gate_id="IST-GATE-07",
    )

    sayac = oversized = cabin_ok = 0
    rng = random.Random(7)
    seviye_gecmisleri = []
    t_start = time.perf_counter()
    for _ in range(180):
        bagaj_tipi = rng.choices(["oversized", "cabin_ok", "personal"], weights=[0.15, 0.65, 0.20])[0]
        if bagaj_tipi == "oversized":
            oversized += 1
            sayac += 1
        elif bagaj_tipi == "cabin_ok":
            cabin_ok += 1
            sayac += 1
        # personal item overhead hesabına dahil edilmez.
        cikti = fusion.guncelle(
            ucus=ucus,
            tahmin=tahmin_riskli,
            toplam_sayilan=sayac,
            oversized_sayisi=oversized,
            cabin_ok_sayisi=cabin_ok,
        )
        action.yayinla(cikti)
        seviye_gecmisleri.append(cikti.aksiyon.seviye)

    avg_per_yolcu = ((time.perf_counter() - t_start) * 1000) / 180
    son = fusion.son_cikti("IST-GATE-07", "TK-2002")
    assert son is not None
    son.aksiyon.override_et("Kaptan karari: devam et")

    assert son.aksiyon.insan_override
    assert audit_obs.log_boyutu == 180
    assert len(central_obs.tum_gate_durumlari) >= 1
    assert avg_per_yolcu < 2.0
    assert any(s in (DolulukSeviyesi.WARNING, DolulukSeviyesi.CRITICAL) for s in seviye_gecmisleri)


# ─────────────────────────────────────────────
# TEST 7: Çok gate eş zamanlı
# ─────────────────────────────────────────────
def test_cok_gate():
    print("\n[TEST 7] Cok Gate Es Zamanli Simulasyon")
    n_gate = 10
    n_frame = 200
    services = []
    for i in range(n_gate):
        calib_service = CalibrationService.create("auto")
        params = GatePhysicalParams(
            gate_id=f"IST-GATE-{i+1:02d}",
            camera_height_m=3.0 + i * 0.1,
            camera_tilt_deg=10.0,
            camera_fov_horizontal_deg=88.0,
            camera_fov_vertical_deg=58.0,
            frame_width_px=1280,
            frame_height_px=720,
        )
        calib = calib_service.calibrate_gate(params)
        svc = InferenceService(
            gate_id=params.gate_id,
            detector=MockDetector(seed=i, avg_detections=3),
            tracker=SimpleTracker(),
            calibration=calib,
        )
        svc.baslat()
        services.append(svc)

    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    t_start = time.perf_counter()
    toplam_frame = 0
    for _ in range(n_frame):
        for svc in services:
            svc.process_frame(dummy)
            toplam_frame += 1
    avg_ms = ((time.perf_counter() - t_start) * 1000) / toplam_frame

    print(f"  ✓ {n_gate} gate x {n_frame} frame = {toplam_frame} toplam frame")
    print(f"  ✓ Ortalama: {avg_ms:.2f}ms/frame")
    assert avg_ms < 20.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
