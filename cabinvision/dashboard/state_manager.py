# dashboard/state_manager.py
#
# KAVRAM: Singleton Pattern (hafif)
# Dashboard her rerun'da Python'u baştan çalıştırır.
# Streamlit session_state bu sorunu çözer — sayfa yenilenince korunur.
# StateManager bu session_state'i düzenli bir arayüzle sarmalar.
#
# KAVRAM: Facade Pattern
# Tüm sistemin state'ine tek bir noktadan erişim.
# Dashboard bileşenleri StateManager üzerinden konuşur,
# alttaki servisleri doğrudan çağırmaz.

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import random
import math
from collections import deque
from typing import Optional
import streamlit as st

from edge.models.calibration_models import GatePhysicalParams, ReferenceObject
from edge.services.calibration_service import CalibrationService
from edge.services.inference_service import (
    InferenceService, MockDetector, SimpleTracker, AdaptiveThreshold
)
from central.models.flight_models import (
    UcusBilgisi, UcakTipi, DolulukSeviyesi, FusionCiktisi
)
from central.repositories.flight_memory_repository import (
    FlightMemoryRepository, UcusKaydi
)
from central.services.prediction_engine import PredictionEngine
from central.services.fusion_engine import (
    FusionEngine, KuralBazliFusion,
    ActionService, GateDashboardObserver,
    CentralDashboardObserver, AuditLogObserver,
)

# Demo varsayımı: personal item olasılığı gerçek operasyonel katsayı değildir.
# Pilot veriyle kalibre edilene kadar yalnızca simülasyon çeşitliliği için kullanılır.
DEMO_PERSONAL_ITEM_PROBABILITY = 0.75


def _init_state():
    """Session state'i başlat — sadece ilk açılışta çalışır."""
    defaults = {
        # Sistem bileşenleri
        "memory_repo":    None,
        "action_service": None,
        "central_obs":    None,
        "audit_obs":      None,
        "gate_services":  {},   # gate_id → InferenceService
        "gate_engines":   {},   # gate_id → FusionEngine
        "gate_tahmins":   {},   # gate_id → DolulukTahmini
        "gate_ucuslar":   {},   # gate_id → UcusBilgisi
        "pred_engine":    None,

        # Simülasyon state
        "sim_calisıyor":  False,
        "sim_yolcu_no":   {},   # gate_id → int
        "sim_bagaj_rng":  {},   # gate_id → Random

        # Grafik geçmişi
        "doluluk_gecmis": {},   # gate_id → deque[(yolcu_no, doluluk)]
        "alert_gecmis":   [],   # list[dict]
        "seviye_gecmis":  {},   # gate_id → deque[str] — timeline için

        # Seçili gate (gate dashboard için)
        "secili_gate":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_gate_calibration_info(gate_id: str) -> Optional[dict]:
    """
    Gate'in kalibrasyon bilgisini döner — dashboard görselleştirmesi için.

    NEDEN AYRI FONKSİYON?
    InferenceService kalibrasyonu private (_calibration) tutuyor.
    Bu fonksiyon dashboard'un ihtiyacı olan bilgiyi okur,
    InferenceService'in iç yapısını dashboard'a sızdırmaz.
    """
    inf = st.session_state["gate_services"].get(gate_id)
    if inf is None:
        return None

    calib = inf._calibration  # InferenceService'in calibration referansı
    return {
        "gate_id":           calib.gate_id,
        "version":           calib.version,
        "method":            calib.calibration_method,
        "cm_per_pixel":      calib.cm_per_pixel_at_ref_distance,
        "confidence":        calib.confidence_score,
        "personal_max_px":   calib.size_thresholds_px.get("personal_item_max", 0),
        "cabin_ok_max_px":   calib.size_thresholds_px.get("cabin_ok_max", 0),
        "notes":             calib.notes,
    }


def get_hat_istatistigi(hat: str) -> Optional[dict]:
    """
    Sefer hafızasından hat istatistiğini döner.
    Dashboard'da "geçmiş pattern" gösterimi için.
    """
    mem = st.session_state.get("memory_repo")
    if mem is None:
        return None

    ist = mem.hat_istatistigi_al(hat)
    if ist is None:
        return None

    return {
        "hat":              ist.hat,
        "kayit_sayisi":     ist.kayit_sayisi,
        "ort_doluluk":      ist.ort_doluluk,
        "ort_oversized":    ist.ort_oversized_oran,
        "guven_skoru":      ist.guven_skoru,
        "max_doluluk":      ist.max_doluluk,
    }


def get_or_init_system(gate_ids: list[str]) -> None:
    """
    Sistemi ilk kez başlat veya gate ekle.
    Idempotent: birden fazla çağrılabilir.
    """
    _init_state()

    # Memory ve engine'ler
    if st.session_state["memory_repo"] is None:
        mem = FlightMemoryRepository(window_size=30)
        _seed_memory(mem)  # Demo için geçmiş veri ekle

        pred = PredictionEngine.create("rule_based", memory=mem)
        action = ActionService()
        central_obs = CentralDashboardObserver()
        audit_obs   = AuditLogObserver()
        action.subscribe(central_obs)
        action.subscribe(audit_obs)

        st.session_state["memory_repo"]    = mem
        st.session_state["pred_engine"]    = pred
        st.session_state["action_service"] = action
        st.session_state["central_obs"]    = central_obs
        st.session_state["audit_obs"]      = audit_obs

    # Her gate için servis
    for gate_id in gate_ids:
        if gate_id not in st.session_state["gate_services"]:
            _init_gate(gate_id)


def kamera_parametresi_guncelle(
    gate_id: str,
    camera_height_m: float,
    camera_tilt_deg: float,
    camera_fov_horizontal_deg: float,
) -> None:
    """
    Kamera parametrelerini değiştirip kalibrasyonu CANLI yeniden hesaplar.

    NEDEN ÖNEMLİ?
    Bu fonksiyon AutoCalibrationStrategy'nin gerçekten parametreye duyarlı
    olduğunu kanıtlar. "Kamerayı farklı bir gate'e taktım, yükseklik/açı
    girdim, sistem kendi kendine ayarlandı" senaryosunu somutlaştırır.

    Mevcut InferenceService'in calibration referansını güncelliyoruz —
    inference_service.update_calibration() zaten bunun için yazılmıştı,
    burada ilk kez gerçek kullanım alanı buluyor.
    """
    inf = st.session_state["gate_services"].get(gate_id)
    if inf is None:
        return

    params = GatePhysicalParams(
        gate_id=gate_id,
        camera_height_m=camera_height_m,
        camera_tilt_deg=camera_tilt_deg,
        camera_fov_horizontal_deg=camera_fov_horizontal_deg,
        camera_fov_vertical_deg=camera_fov_horizontal_deg * 0.65,  # tipik oran
        frame_width_px=1280,
        frame_height_px=720,
    )

    calib_service = CalibrationService.create("auto")
    yeni_kalibrasyon = calib_service.calibrate_gate(params, force_recalibrate=True)

    inf.update_calibration(yeni_kalibrasyon)

    # Kullanıcının son girdiği parametreleri sakla — slider'lar bir sonraki
    # rerun'da bu değerlerden başlasın
    st.session_state.setdefault("kamera_params", {})
    st.session_state["kamera_params"][gate_id] = {
        "height": camera_height_m,
        "tilt":   camera_tilt_deg,
        "fov":    camera_fov_horizontal_deg,
    }


def get_kamera_parametresi(gate_id: str) -> dict:
    """Mevcut/son kullanılan kamera parametrelerini döner — slider başlangıç değeri için."""
    varsayilan = {"height": 3.2, "tilt": 12.0, "fov": 88.0}
    return st.session_state.get("kamera_params", {}).get(gate_id, varsayilan)


def _init_gate(gate_id: str) -> None:
    """Tek gate için tüm servisleri başlat."""
    seed = hash(gate_id) % 10000

    params = GatePhysicalParams(
        gate_id=gate_id,
        camera_height_m=3.2 + (seed % 10) * 0.08,
        camera_tilt_deg=12.0,
        camera_fov_horizontal_deg=88.0,
        camera_fov_vertical_deg=58.0,
        frame_width_px=1280,
        frame_height_px=720,
    )
    ref = ReferenceObject(55.0, 40.0, (380, 280, 600, 500), 2.0)
    calib = CalibrationService.create("manual").calibrate_gate(params, ref)

    detector = MockDetector(seed=seed, avg_detections=3)
    tracker  = SimpleTracker()
    inf_svc  = InferenceService(gate_id, detector, tracker, calib)
    inf_svc.baslat()

    fusion = FusionEngine(KuralBazliFusion())

    # Gate observer
    gate_obs = GateDashboardObserver(gate_id)
    st.session_state["action_service"].subscribe(gate_obs)

    st.session_state["gate_services"][gate_id] = inf_svc
    st.session_state["gate_engines"][gate_id]  = fusion
    st.session_state["sim_yolcu_no"][gate_id]  = 0
    st.session_state["sim_bagaj_rng"][gate_id] = random.Random(seed + 1)
    st.session_state["doluluk_gecmis"][gate_id] = deque(maxlen=200)
    st.session_state["seviye_gecmis"][gate_id]  = deque(maxlen=200)

    # Demo uçuş
    ucus = _demo_ucus(gate_id, seed)
    st.session_state["gate_ucuslar"][gate_id] = ucus

    pred = st.session_state["pred_engine"]
    st.session_state["gate_tahmins"][gate_id] = pred.tahmin_uret(ucus)


def sim_adim(gate_id: str) -> Optional[FusionCiktisi]:
    """
    Bir boarding adımı simüle et.
    Her çağrıda bir yolcu geçer, bagajı işlenir.

    YOL HARİTASI MADDE C2 — kritik düzeltme:
    Eski davranış: HER yolcuya otomatik bir bagaj atanıyordu (beyan oranı,
    sefer hafızası, hiçbir gerçek dünya varsayımı dikkate alınmıyordu).
    Bu, PredictionEngine'in kullandığı mantıkla taban tabana zıttı — iki
    modül birbirinden habersiz farklı "gerçek dünya modeli" kullanıyordu.
    Bu tutarsızlık, %553 doluluk hatasının ikinci kök nedeniydi (C1 ile
    birlikte, slider'ın uçak kapasitesinden bağımsız olmasıyla).

    Yeni davranış: Her yolcu için ÖNCE "bu yolcu kabin bagajı getirir mi?"
    olasılığı hesaplanır. Bu olasılık artık sabit %100 (eskisi gibi) değil,
    PredictionEngine.KuralBazliStrateji'nin kullandığı AYNI sinyal ailesini
    kullanır: sefer hafızası ağırlıklı tahmin (bkz. Madde A4/A5). Böylece
    simülasyon motoru ile tahmin motoru artık aynı varsayımı paylaşıyor —
    "tek ortak domain modeli" prensibi (3 bağımsız AI denetiminin ortak
    önerisiydi).

    BİLİNÇLİ KISIT (Gemini 3. tur, madde 2 — kabul edildi, gerekçesiyle):
    Her yolcu için BAĞIMSIZ bir rastgele deneme yapılıyor (rng.random() <
    olasılık). Gerçek havacılıkta aynı PNR/rezervasyon altında seyahat eden
    aile/grup üyelerinin bagaj davranışı KORELASYONLUDUR (örnek: 4 kişilik
    bir aile genelde 2 büyük valiz + 2 sırt çantası şeklinde dağıtır, biri
    valiz getirirse diğerinin getirme olasılığı bağımsız değildir). PNR bazlı
    grup korelasyonu modellemek bu prototipin kapsamı için "aşırı mühendislik"
    (over-engineering) olarak değerlendirildi ve bilinçli olarak kapsam dışı
    bırakıldı. Bağımsız Bernoulli denemeleri matematiksel bir basitleştirme
    olarak kabul edilmiştir — gizlenmiyor, burada açıkça belgeleniyor.
    """
    ucus   = st.session_state["gate_ucuslar"].get(gate_id)
    tahmin = st.session_state["gate_tahmins"].get(gate_id)
    inf    = st.session_state["gate_services"].get(gate_id)
    fusion = st.session_state["gate_engines"].get(gate_id)
    rng    = st.session_state["sim_bagaj_rng"].get(gate_id)

    if not all([ucus, tahmin, inf, fusion, rng]):
        return None

    yolcu_no = st.session_state["sim_yolcu_no"][gate_id]
    if yolcu_no >= ucus.toplam_yolcu:
        return None  # Boarding bitti

    # ── C2 (GEMINI DÜZELTMESİ İLE DÜZELTİLDİ): Bagaj getirme olasılığı ──
    #
    # ÖNCEKİ HATALI VERSİYON (bağımsız denetimde bulundu):
    #   bagaj_getirme_olasiligi = tahmin.tahmini_doluluk_orani
    # Bu YANLIŞTI çünkü tahmini_doluluk_orani = toplam_bagaj / KAPASİTE'dir,
    # yolcu sayısına bölünmüş bir olasılık DEĞİLDİR. Örnek (Gemini'nin
    # bulduğu somut kanıt): 150 yolcu, kapasite 120, tahmin 90 bagaj ise
    # oran %75 çıkar; bu %75'i yolcu başına olasılık sanıp uygularsak
    # 150*0.75=112.5 bagaj üretilir — tahmin edilen 90'dan %25 fazla.
    # Yani "tutarlılık" sağlamaya çalışırken yeni bir sapma sokulmuş oldu.
    #
    # DOĞRU FORMÜL: olasılık = tahmini_toplam_bagaj / toplam_yolcu
    # (DolulukTahmini.tahmini_toplam_bagaj alanı bu düzeltme için eklendi,
    # bkz. central/models/flight_models.py ve central/services/prediction_engine.py)
    # Slider degerlerinden gelen check-in beyan orani dogrudan kullaniliyor.
    # Onceki versiyon sefer hafizasina bakiyordu (tahmini_toplam_bagaj/yolcu),
    # bu da slider %95'e cekilse bile simulasyonun dusuk olasilik uretmesine
    # sebep oluyordu. Artik ucus.cabin_beyan_sayisi (slider'dan geliyor) /
    # toplam_yolcu formulu kullaniliyor -- slider ne gosteriyorsa simule ediyor.
    beyan_orani = ucus.cabin_beyan_sayisi / max(ucus.toplam_yolcu, 1)
    bagaj_getirme_olasiligi = min(beyan_orani, 1.0)

    yolcu_no += 1
    st.session_state["sim_yolcu_no"][gate_id] = yolcu_no

    # ── GEMINI 5. TUR MADDE 2 DÜZELTMESİ (kabul edildi — kritik düzeltme) ──
    #
    # ÖNCEKİ HATALI VARSAYIM (4. turda "ek kod gerekmedi" diye yanlış
    # kapatılmıştı — bu değerlendirme YANLIŞTI, geri alınıyor):
    # Yolcu "oversized YA DA cabin_ok YA DA personal" arasında KARŞILIKLI
    # DIŞLAYAN (mutually exclusive) bir seçime zorlanıyordu. THY kuralı
    # (1 cabin bagaj + 1 personal item, AYNI ANDA) burada çiğneniyordu:
    # yolcu %30 ihtimalle "personal" seçtiğinde, o yolcunun cabin_ok/
    # oversized getirme ihtimali SIFIRLANIYORDU — oysa gerçekte ikisini
    # birden getirebilirdi. Bu, Baş Üstü Dolabı doluluğunun SİSTEMATİK
    # OLARAK EKSİK TAHMİN EDİLMESİNE (under-prediction) yol açıyordu —
    # kapasite aşımı riskini KAÇIRMA potansiyeli taşıyan, güvenli olmayan
    # bir yöndeki hataydı.
    #
    # DÜZELTME: İki BAĞIMSIZ olasılık denemesi (Bernoulli) yapılıyor:
    #   1) Bu yolcu Baş Üstü Dolabı'na bagaj getirir mi?
    #   2) Bu yolcu (BAĞIMSIZ OLARAK) Koltuk Altı'na personal item getirir mi?
    # İkisi de olabilir, ikisi de olmayabilir, sadece biri olabilir — THY'nin
    # 1+1 kuralını artık doğru modelliyor.

    from edge.models.detection_models import BoyutSinifi

    # 1) BAŞ ÜSTÜ DOLABI — mevcut tahmin motoru sinyali (zaten doğru formül)
    bas_ustu_getirildi = rng.random() < bagaj_getirme_olasiligi
    if bas_ustu_getirildi:
        manuel_oranlar = st.session_state.get("manuel_oversized_oran", {})
        # Slider oversized oranini kullan (varsayilan 0.15 degil, gercek deger)
        over_w = manuel_oranlar.get(gate_id, ucus.oversized_beyan / max(ucus.cabin_beyan_sayisi, 1))
        bas_ustu_turu = (
            BoyutSinifi.OVERSIZED if rng.random() < over_w else BoyutSinifi.CABIN_OK
        )
        inf._InferenceService__sayilan_idler[yolcu_no + 9000] = bas_ustu_turu

    # 2) KOLTUK ALTI — AYRI/BAĞIMSIZ bir olasılık. Bu sistem için literatürde
    # doğrulanmış bir oran bulunamadı (bkz. THY check-in akışı araştırması,
    # Madde A4) — bu yüzden burada kullanılan PERSONAL_ITEM_GETIRME_ORANI
    # KANITSIZ bir varsayımdır, dürüstçe işaretleniyor: çoğu yolcunun en
    # azından küçük bir kişisel eşya (laptop çantası, el çantası) taşıdığı
    # makul ama doğrulanmamış bir sezgiye dayanıyor. Gerçek pilot veriyle
    # kalibre edilmesi gerekir.
    koltuk_alti_getirildi = rng.random() < DEMO_PERSONAL_ITEM_PROBABILITY
    if koltuk_alti_getirildi:
        # +19000: bas_ustu'nun +9000 id aralığıyla ÇAKIŞMASIN diye farklı aralık
        inf._InferenceService__sayilan_idler[yolcu_no + 19000] = BoyutSinifi.PERSONAL_ITEM
    # else: yolcu bagaj getirmedi, sayaca hiçbir şey eklenmiyor —
    # bu, eski "her yolcu bagaj getirir" varsayımının düzeltilmiş hali.

    dagilim  = inf.boyut_dagilimi
    cikti    = fusion.guncelle(
        ucus=ucus,
        tahmin=tahmin,
        # GEMINI 3. TUR MADDE 3 (B): overhead bin doluluğu artık personal
        # item HARİÇ hesaplanıyor — THY kuralına göre personal item koltuk
        # altına gider, dolaba değil. Eskiden inf.toplam_sayilan kullanılıyordu
        # (personal item dahildi) — bu fazla-sayım hatasıydı.
        toplam_sayilan=inf.overhead_bin_sayilan,
        oversized_sayisi=dagilim.get("oversized", 0),
        cabin_ok_sayisi=dagilim.get("cabin_ok", 0),
    )
    st.session_state["action_service"].yayinla(cikti)

    # Geçmiş güncelle
    st.session_state["doluluk_gecmis"][gate_id].append(
        (yolcu_no, cikti.gercek_doluluk * 100)
    )
    st.session_state["seviye_gecmis"][gate_id].append(cikti.aksiyon.seviye.value)

    return cikti


def yeni_ucus_baslat(gate_id: str) -> None:
    """Gate için yeni uçuş başlat — sayaçları sıfırla."""
    inf = st.session_state["gate_services"].get(gate_id)
    if inf:
        inf.yeni_ucus_baslat()

    seed = hash(gate_id + str(time.time())) % 10000
    ucus = _demo_ucus(gate_id, seed)
    pred = st.session_state["pred_engine"]
    pred.tahmin_gecersiz_kil(ucus.ucus_no)

    st.session_state["gate_ucuslar"][gate_id]   = ucus
    st.session_state["gate_tahmins"][gate_id]   = pred.tahmin_uret(ucus)
    st.session_state["sim_yolcu_no"][gate_id]   = 0
    st.session_state["sim_bagaj_rng"][gate_id]  = random.Random(seed)
    st.session_state["doluluk_gecmis"][gate_id] = deque(maxlen=200)
    st.session_state["seviye_gecmis"][gate_id]  = deque(maxlen=200)


def ucus_parametre_guncelle(
    gate_id: str,
    toplam_yolcu: int,
    beyan_orani_pct: int,
    oversized_oran_pct: int,
    ucak_tipi_str: str,
) -> None:
    """
    Kullanıcının slider'la girdiği parametrelerle yeni uçuş kurar.

    NEDEN AYRI FONKSİYON?
    _demo_ucus() rastgele üretiyor. Bu fonksiyon kullanıcı kontrollü.
    İkisi farklı senaryolar: biri otomatik demo, biri manuel test.
    Var olan _demo_ucus'a dokunmuyoruz — yeni davranış yeni fonksiyonda.
    """
    inf = st.session_state["gate_services"].get(gate_id)
    if inf:
        inf.yeni_ucus_baslat()

    tip_map = {
        "Dar Gövde (A320/A321)": UcakTipi.NARROW_BODY,
        "Geniş Gövde (B777/A350)": UcakTipi.WIDE_BODY,
    }
    ucak_tipi = tip_map.get(ucak_tipi_str, UcakTipi.NARROW_BODY)

    # YOL HARİTASI MADDE A6: int() yerine math.ceil() — risk hesaplarında
    # eksik tahmin fazla tahminden daha zararlı, bu yüzden yukarı yuvarlanır.
    beyan_sayisi = math.ceil(toplam_yolcu * beyan_orani_pct / 100)
    oversized_sayisi = math.ceil(beyan_sayisi * oversized_oran_pct / 100)

    ucus = UcusBilgisi(
        ucus_no=f"TK-{random.randint(1000,9999)}",
        hat="IST-DXB",  # manuel modda sabit hat — sefer hafızasıyla tutarlı kalır
        ucak_tipi=ucak_tipi,
        toplam_yolcu=toplam_yolcu,
        cabin_beyan_sayisi=beyan_sayisi,
        oversized_beyan=oversized_sayisi,
        gate_id=gate_id,
        manuel_senaryo=True,
    )

    pred = st.session_state["pred_engine"]
    pred.tahmin_gecersiz_kil(ucus.ucus_no)

    st.session_state["gate_ucuslar"][gate_id]   = ucus
    st.session_state["gate_tahmins"][gate_id]   = pred.tahmin_uret(ucus)
    st.session_state["sim_yolcu_no"][gate_id]   = 0
    st.session_state["sim_bagaj_rng"][gate_id]  = random.Random(
        hash((gate_id, toplam_yolcu, oversized_oran_pct))
    )
    st.session_state["doluluk_gecmis"][gate_id] = deque(maxlen=200)
    st.session_state["seviye_gecmis"][gate_id]  = deque(maxlen=200)

    # sim_adim fonksiyonu bagaj türü dağılımını rng'den seçiyor,
    # oversized oranını etkilemek için ağırlıkları da güncelliyoruz
    st.session_state.setdefault("manuel_oversized_oran", {})
    st.session_state["manuel_oversized_oran"][gate_id] = oversized_oran_pct / 100


def _demo_ucus(gate_id: str, seed: int) -> UcusBilgisi:
    rng = random.Random(seed)
    hatlar = ["IST-DXB", "IST-LHR", "IST-JFK", "IST-AYT", "IST-FRA"]
    hat    = rng.choice(hatlar)
    yolcu  = rng.randint(120, 185)
    beyan  = int(yolcu * rng.uniform(0.40, 0.80))
    over   = rng.randint(3, 18)
    tip    = rng.choice([UcakTipi.NARROW_BODY, UcakTipi.NARROW_BODY, UcakTipi.WIDE_BODY])
    ucus_no = f"TK-{rng.randint(1000, 9999)}"

    return UcusBilgisi(
        ucus_no=ucus_no,
        hat=hat,
        ucak_tipi=tip,
        toplam_yolcu=yolcu,
        cabin_beyan_sayisi=beyan,
        oversized_beyan=over,
        gate_id=gate_id,
    )


def _seed_memory(mem: FlightMemoryRepository) -> None:
    """
    Demo için geçmiş sefer verisi ekle.

    GEMINI 4. TUR MADDE 1 DÜZELTMESİ (kabul edildi):
    Önceki versiyonda `doluluk_orani` tamamen BAĞIMSIZ rastgele üretiliyordu
    (rng.uniform(0.55, 0.98)) — oversized_sayisi/cabin_ok_sayisi'nden hiç
    türetilmiyordu. Bu, B düzeltmesinden (overhead_bin_sayilan, personal
    item hariç) SONRA tutarsız hale geldi. Düzeltme: doluluk_orani artık
    (oversized_sayisi + cabin_ok_sayisi) / kapasite formülünden TÜRETİLİYOR.

    GEMINI 5. TUR MADDE 3 DÜZELTMESİ (kabul edildi):
    Önceki versiyon TÜM kayıtları sabit narrow_body (kapasite=120)
    varsayımıyla üretiyordu — gerçekçi değildi, çünkü gerçekte bir hat hem
    dar hem geniş gövde uçakla uçabilir. Düzeltme: her hat için KARIŞIK uçak
    tipi dağılımı üretiliyor (her kayıt kendi gerçek kapasitesine göre
    doluluk hesaplıyor). Bu, FlightMemoryRepository.hat_istatistigi_al()'ın
    yeni uçak-tipi-filtreleme özelliğinin (bkz. central/repositories/
    flight_memory_repository.py) gerçek bir etkisi olmasını sağlıyor —
    aksi halde tüm kayıtlar zaten aynı tipte olsaydı filtreleme hiçbir şeyi
    değiştirmezdi, test edilemez kalırdı.
    """
    rng = random.Random(42)
    hatlar = ["IST-DXB", "IST-LHR", "IST-JFK", "IST-AYT", "IST-FRA"]

    for i in range(40):
        hat = hatlar[i % 5]

        # Karışık uçak tipi — %70 narrow, %30 wide (gerçekçi bir hat profili)
        tip = rng.choices(
            [UcakTipi.NARROW_BODY, UcakTipi.WIDE_BODY],
            weights=[0.70, 0.30]
        )[0]
        kapasite = tip.kapasite  # ARTIK SABİT DEĞİL — her kaydın kendi gerçek kapasitesi

        toplam = rng.randint(60, 110)   # toplam tespit edilen NESNE (yolcu sayısı değil)
        oversized = int(toplam * rng.uniform(0.10, 0.25))
        cabin_ok  = int(toplam * 0.65)
        personal  = toplam - oversized - cabin_ok

        # doluluk_orani SADECE oversized+cabin_ok'tan, kendi kapasitesine göre
        overhead_toplam = oversized + cabin_ok
        doluluk = min(overhead_toplam / kapasite, 1.0)

        mem.kayit_ekle(UcusKaydi(
            ucus_no=f"TK-DEMO-{i}",
            hat=hat,
            toplam_bagaj=toplam,
            oversized_sayisi=oversized,
            cabin_ok_sayisi=cabin_ok,
            personal_sayisi=personal,
            doluluk_orani=doluluk,
            ucak_tipi=tip.value,   # ARTIK GERÇEK TİP — sabit "narrow_body" değil
        ))
