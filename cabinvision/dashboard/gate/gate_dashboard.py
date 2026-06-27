# dashboard/gate/gate_dashboard.py
#
# Gate personelinin ekranı.
# Tek gate, gerçek zamanlı doluluk, aksiyon önerisi.
# Streamlit'te çalıştır: streamlit run dashboard/gate/gate_dashboard.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import deque

from dashboard.state_manager import (
    get_or_init_system, sim_adim, yeni_ucus_baslat,
    get_gate_calibration_info, get_hat_istatistigi,
    ucus_parametre_guncelle,
    kamera_parametresi_guncelle, get_kamera_parametresi,
)
from central.models.flight_models import DolulukSeviyesi, UcakTipi

# ── SAYFA AYARLARI ──
st.set_page_config(
    page_title="CabinVision | Gate Monitor",
    page_icon="✈",
    layout="wide",
)

# ── CSS ──
st.markdown("""
<style>
*{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.stApp{background:#0d1117;color:#f0f4f8}
[data-testid="metric-container"]{
    background:#161b22;border:1px solid #2d3748;
    border-radius:10px;padding:14px;
}
.block-container{padding-top:1.5rem}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #2d3748}

/* Aksiyon banner renkleri */
.banner-normal{
    background:rgba(72,187,120,0.08);border-left:4px solid #48bb78;
    border-radius:8px;padding:14px 18px;margin:8px 0;
}
.banner-warning{
    background:rgba(246,173,85,0.10);border-left:4px solid #f6ad55;
    border-radius:8px;padding:14px 18px;margin:8px 0;
    animation:pulse-warn 2s infinite;
}
.banner-critical{
    background:rgba(229,62,62,0.10);border-left:4px solid #e53e3e;
    border-radius:8px;padding:14px 18px;margin:8px 0;
    animation:pulse-crit 1.2s infinite;
}
@keyframes pulse-warn{0%,100%{box-shadow:0 0 0 0 rgba(246,173,85,0)}50%{box-shadow:0 0 0 6px rgba(246,173,85,0.08)}}
@keyframes pulse-crit{0%,100%{box-shadow:0 0 0 0 rgba(229,62,62,0)}50%{box-shadow:0 0 0 8px rgba(229,62,62,0.12)}}

.tag{display:inline-block;padding:2px 9px;border-radius:12px;font-size:11px;font-weight:600;margin-right:4px}
.tag-r{background:rgba(229,62,62,0.15);color:#e53e3e;border:1px solid #9b2335}

/* ── ÖZGÜNLÜK 1: Kalibrasyon Paneli — BELİRGİN VERSİYON ── */
.calib-card{
    background:linear-gradient(135deg,rgba(56,189,248,0.10) 0%,rgba(167,139,250,0.06) 100%);
    border:1.5px solid #38bdf8;border-radius:12px;
    padding:20px 22px;margin:14px 0;
    box-shadow:0 0 24px rgba(56,189,248,0.08);
}
.calib-header{
    display:flex;align-items:center;gap:10px;margin-bottom:14px;
    padding-bottom:12px;border-bottom:1px solid rgba(56,189,248,0.25);
}
.calib-title{font-size:15px;font-weight:700;color:#f0f4f8}
.calib-badge{
    font-family:monospace;font-size:11px;font-weight:700;
    background:rgba(56,189,248,0.18);color:#38bdf8;
    border:1px solid #38bdf8;border-radius:5px;padding:3px 10px;
    letter-spacing:0.5px;
}
.calib-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 20px}
.calib-row{display:flex;justify-content:space-between;padding:6px 0;
    border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px}
.calib-label{color:#9aa5b4}
.calib-val{font-family:monospace;color:#f0f4f8;font-weight:700;font-size:14px}
.calib-confidence-bar{height:8px;background:#1c2330;border-radius:4px;margin-top:10px;overflow:hidden}
.calib-confidence-fill{height:100%;border-radius:4px;transition:width .4s ease}

/* ── ÖZGÜNLÜK: Senaryo Parametre Paneli ── */
.scenario-card{
    background:rgba(229,62,62,0.04);border:1px solid #2d3748;
    border-radius:12px;padding:18px 20px;margin:14px 0;
}
.scenario-header{
    display:flex;align-items:center;gap:8px;margin-bottom:4px;
    font-size:14px;font-weight:700;color:#f0f4f8;
}

/* ── ÖZGÜNLÜK 2: Sefer Hafızası Kartı ── */
.memory-card{
    background:rgba(167,139,250,0.06);border:1px solid #6d4f9e;
    border-radius:10px;padding:14px 18px;margin:10px 0;
}
.memory-header{
    display:flex;align-items:center;gap:6px;
    color:#a78bfa;font-weight:600;font-size:13px;margin-bottom:8px;
}
.memory-stat{display:inline-block;margin-right:18px;font-size:13px}
.memory-stat b{color:#f0f4f8}

/* ── ÖZGÜNLÜK 4: Aksiyon Timeline ── */
.timeline-wrap{display:flex;align-items:center;gap:2px;margin:10px 0;height:32px}
.timeline-seg{flex:1;height:10px;border-radius:2px;transition:opacity .2s}
.timeline-seg:hover{opacity:0.7}
.timeline-legend{display:flex;gap:16px;margin-top:6px;font-size:11px;color:#8892a4}
.timeline-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.tag-g{background:rgba(72,187,120,0.12);color:#48bb78}
.tag-a{background:rgba(246,173,85,0.12);color:#f6ad55}
.tag-m{background:#1c2330;color:#8892a4}
</style>
""", unsafe_allow_html=True)

# ── SİSTEM BAŞLAT ──
GATE_IDS = [
    "IST-GATE-07", "IST-GATE-12", "IST-GATE-18",
    "IST-GATE-23", "IST-GATE-31",
]
get_or_init_system(GATE_IDS)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ✈ CabinVision")
    st.markdown("**Gate Monitor**")
    st.divider()

    gate_id = st.selectbox(
        "Gate Seç",
        GATE_IDS,
        key="secili_gate_select",
    )
    st.session_state["secili_gate"] = gate_id

    ucus = st.session_state["gate_ucuslar"].get(gate_id)
    if ucus:
        st.markdown(f"**Uçuş:** `{ucus.ucus_no}`")
        st.markdown(f"**Hat:** {ucus.hat}")
        st.markdown(f"**Yolcu:** {ucus.toplam_yolcu}")
        st.markdown(f"**Uçak:** {ucus.ucak_tipi.value}")
        st.markdown(f"**Kapasite:** {ucus.dolap_kapasitesi}")

    st.divider()

    sim_hizi = st.select_slider(
        "Simülasyon Hızı",
        options=["Yavaş", "Normal", "Hızlı"],
        value="Normal",
    )
    hiz_map = {"Yavaş": 0.35, "Normal": 0.12, "Hızlı": 0.03}
    bekleme = hiz_map[sim_hizi]

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Başlat", use_container_width=True, type="primary"):
            st.session_state["sim_calisıyor"] = True
    with col2:
        if st.button("⏸ Durdur", use_container_width=True):
            st.session_state["sim_calisıyor"] = False

    if st.button("🔄 Yeni Uçuş", use_container_width=True):
        yeni_ucus_baslat(gate_id)
        st.session_state["sim_calisıyor"] = False
        st.rerun()


# ── BAŞLIK ──
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"## Gate {gate_id.split('-')[-1]} — Canlı İzleme")
with col_h2:
    durum_renk = "🟢" if st.session_state["sim_calisıyor"] else "⚪"
    st.markdown(f"<br/>{durum_renk} {'CANLI' if st.session_state['sim_calisıyor'] else 'BEKLEMEDE'}", unsafe_allow_html=True)

st.divider()

# ── ÖZGÜNLÜK 1 + SENARYO PANELİ: yan yana, büyük, görünür ──
col_calib, col_scenario = st.columns([1, 1])

with col_calib:
    calib_info = get_gate_calibration_info(gate_id)
    if calib_info:
        conf = calib_info["confidence"]
        conf_renk = "#48bb78" if conf > 0.75 else ("#f6ad55" if conf > 0.5 else "#e53e3e")
        method_tr = "Manuel (referans nesne)" if calib_info["method"] == "manual" else "Otomatik (geometri)"

        st.markdown(f"""
        <div class="calib-card">
            <div class="calib-header">
                <span class="calib-title">📐 Gate Kalibrasyonu</span>
                <span class="calib-badge">{calib_info["method"].upper()}</span>
            </div>
            <div style="font-size:11px;color:#9aa5b4;margin-bottom:10px">{method_tr}</div>
            <div class="calib-grid">
                <div class="calib-row">
                    <span class="calib-label">cm / piksel</span>
                    <span class="calib-val">{calib_info["cm_per_pixel"]:.4f}</span>
                </div>
                <div class="calib-row">
                    <span class="calib-label">Güven skoru</span>
                    <span class="calib-val" style="color:{conf_renk}">%{int(conf*100)}</span>
                </div>
                <div class="calib-row">
                    <span class="calib-label">Personal eşik</span>
                    <span class="calib-val">{calib_info["personal_max_px"]:.0f}px</span>
                </div>
                <div class="calib-row">
                    <span class="calib-label">Cabin OK eşik</span>
                    <span class="calib-val">{calib_info["cabin_ok_max_px"]:.0f}px</span>
                </div>
            </div>
            <div class="calib-confidence-bar">
                <div class="calib-confidence-fill" style="width:{conf*100}%;background:{conf_renk}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Bu gate'e özgü eşikler — kamera açısı, yüksekliği ve "
            "referans nesne ölçümünden otomatik hesaplandı."
        )

        # ── KAMERA PARAMETRESİ — CANLI YENİDEN KALİBRASYON ──
        st.markdown(
            '<div style="font-size:12px;font-weight:600;color:#38bdf8;'
            'margin:14px 0 2px">⚙ Kamera Parametrelerini Değiştir</div>'
            '<div style="font-size:11px;color:#8892a4;margin-bottom:8px">'
            'Farklı bir gate\'e taktığını varsay — değerleri değiştir, '
            'kalibrasyonun anında kendini ayarladığını gör.</div>',
            unsafe_allow_html=True
        )

        mevcut = get_kamera_parametresi(gate_id)
        kp_h = st.slider(
            "Kamera Yüksekliği (m)", 2.0, 8.0, mevcut["height"], 0.1,
            key=f"kp_h_{gate_id}",
        )
        kp_t = st.slider(
            "Kamera Açısı / Tilt (°)", 0.0, 60.0, mevcut["tilt"], 1.0,
            key=f"kp_t_{gate_id}",
        )
        kp_f = st.slider(
            "Yatay Görüş Açısı / FOV (°)", 60.0, 120.0, mevcut["fov"], 1.0,
            key=f"kp_f_{gate_id}",
        )

        if st.button("🔁 Kalibrasyonu Yeniden Hesapla", use_container_width=True):
            kamera_parametresi_guncelle(gate_id, kp_h, kp_t, kp_f)
            st.rerun()

with col_scenario:
    st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="scenario-header">🎛 Senaryo Parametreleri</div>'
        '<div style="font-size:11px;color:#9aa5b4;margin-bottom:14px">'
        'Parametreleri ayarla, "Senaryoyu Uygula" ile yeni uçuş kur.</div>',
        unsafe_allow_html=True
    )

    sim_calisiyor_simdi = st.session_state["sim_calisıyor"]
    if sim_calisiyor_simdi:
        st.info("Boarding devam ederken parametre değiştirilemez. Önce durdurun.")

    # YOL HARİTASI MADDE C1: Slider artık uçak tipinden BAĞIMSIZ değil.
    # Önce uçak tipi seçilir, sonra yolcu slider'ının üst sınırı buna göre
    # otomatik ayarlanır. Kural netleştirildi: alt sınırın koltuk sayısına
    # eşit olması GEREKMEZ (300 koltuklu uçak 200 kişiyle de kalkabilir) —
    # kısıtlanması gereken yalnızca ÜST sınır, çünkü bir uçağa fiziksel
    # koltuk sayısından fazla yolcu binemez. (Bkz. %553 doluluk hatası kökü.)
    p_ucak = st.selectbox(
        "Uçak Tipi",
        ["Dar Gövde (A320/A321)", "Geniş Gövde (B777/A350)"],
        disabled=sim_calisiyor_simdi,
        key=f"p_ucak_{gate_id}",
    )

    _tip_map_ui = {
        "Dar Gövde (A320/A321)": UcakTipi.NARROW_BODY,
        "Geniş Gövde (B777/A350)": UcakTipi.WIDE_BODY,
    }
    _secili_tip = _tip_map_ui[p_ucak]
    _min_koltuk, _max_koltuk = _secili_tip.koltuk_sayisi_araligi

    p_yolcu = st.slider(
        "Toplam Yolcu",
        min_value=20,                # makul düşük alt sınır, koltuk sayısına bağlı değil
        max_value=_max_koltuk,       # FİZİKSEL ÜST SINIR — uçak tipine göre değişir
        value=min(168, _max_koltuk),
        disabled=sim_calisiyor_simdi,
        key=f"p_yolcu_{gate_id}",
        help=f"{p_ucak} için gerçekçi koltuk aralığı: {_min_koltuk}-{_max_koltuk}",
    )
    p_beyan = st.slider(
        "Ön Tahmin Sinyali (PNR/Check-in) (%)", 10, 95, 55, disabled=sim_calisiyor_simdi,
        key=f"p_beyan_{gate_id}",
    )
    p_over = st.slider(
        "Oversized Bagaj Oranı (%)", 0, 50, 15, disabled=sim_calisiyor_simdi,
        key=f"p_over_{gate_id}",
    )

    if st.button(
        "✅ Senaryoyu Uygula", use_container_width=True, type="primary",
        disabled=sim_calisiyor_simdi,
    ):
        ucus_parametre_guncelle(gate_id, p_yolcu, p_beyan, p_over, p_ucak)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ── BOARDING ADIMı ──
if st.session_state["sim_calisıyor"]:
    cikti = sim_adim(gate_id)
    if cikti is None:
        st.session_state["sim_calisıyor"] = False
        st.success("✅ Boarding tamamlandı.")
else:
    cikti = None
    fusion = st.session_state["gate_engines"].get(gate_id)
    ucus_tmp = st.session_state["gate_ucuslar"].get(gate_id)
    if fusion and ucus_tmp:
        cikti = fusion.son_cikti(gate_id, ucus_tmp.ucus_no)

# ── TAHMİN BANNER ──
tahmin = st.session_state["gate_tahmins"].get(gate_id)
if tahmin:
    t_ikon  = "🔴" if tahmin.asim_bekleniyor else ("🟡" if tahmin.tahmini_doluluk_orani > 0.75 else "🟢")
    if tahmin.asim_bekleniyor:
        t_class = "banner-critical"
    elif tahmin.tahmini_doluluk_orani > 0.75:
        t_class = "banner-warning"
    else:
        t_class = "banner-normal"
    st.markdown(
        f'<div class="{t_class}">'
        f'<strong>{t_ikon} TAHMİN (Boarding öncesi):</strong> {tahmin.aciklama}'
        f'</div>',
        unsafe_allow_html=True
    )

# ── ÖZGÜNLÜK 2: SEFER HAFIZASI KARTI ──
# Tahminin neye dayandığını gösterir — check-in verisi + geçmiş hat istatistiği
if ucus:
    hat_ist = get_hat_istatistigi(ucus.hat)
    if hat_ist and hat_ist["kayit_sayisi"] >= 3:
        guven_renk = "#48bb78" if hat_ist["guven_skoru"] > 0.6 else "#f6ad55"
        st.markdown(
            f'<div class="memory-card">'
            f'<div class="memory-header">🧠 SEFER HAFIZASI — {ucus.hat} hattı geçmişi</div>'
            f'<span class="memory-stat">Geçmiş ortalama talep: <b>%{int(hat_ist["ort_doluluk"]*100)}</b></span>'
            f'<span class="memory-stat">Oversized oranı: <b>%{int(hat_ist["ort_oversized"]*100)}</b></span>'
            f'<span class="memory-stat">Veri: <b>{hat_ist["kayit_sayisi"]} uçuş</b></span>'
            f'<span class="memory-stat">Güven: <b style="color:{guven_renk}">%{int(hat_ist["guven_skoru"]*100)}</b></span>'
            f'<div style="font-size:11px;color:#8892a4;margin-top:6px">'
            f'Bu hattın geçmiş verisi, ön tahmin sinyaliyle harmanlanarak yukarıdaki tahmine dahil edildi.'
            f'</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="memory-card" style="opacity:0.6">'
            f'<div class="memory-header">🧠 SEFER HAFIZASI — {ucus.hat} hattı</div>'
            f'<span style="font-size:12px;color:#8892a4">Henüz yeterli geçmiş veri yok (min. 3 uçuş gerekli). '
            f'Tahmin yalnızca ön tahmin sinyaline dayanıyor.</span>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── METRİKLER ──
yolcu_no  = st.session_state["sim_yolcu_no"].get(gate_id, 0)
inf       = st.session_state["gate_services"].get(gate_id)
dagilim   = inf.boyut_dagilimi if inf else {}
# GEMINI 3. TUR MADDE 3 (B): Kabin doluluğu SADECE overhead bin'i kullanan
# bagajlardan (cabin_ok + oversized) hesaplanır — personal item koltuk
# altına gittiği için dolap doluluğunu etkilemez.
overhead_sayilan = inf.overhead_bin_sayilan if inf else 0
personal_sayilan = dagilim.get("personal_item", 0)
ucus      = st.session_state["gate_ucuslar"].get(gate_id)
kapasite  = ucus.dolap_kapasitesi if ucus else 120
doluluk   = round((overhead_sayilan / kapasite) * 100, 1) if kapasite > 0 else 0

# GEMINI 4. TUR MADDE 2 DÜZELTMESİ (kabul edildi — UX netliği):
# Eskiden "Toplam Bagaj: 140" ve "Baş Üstü Dolap Talebi: %83" yan yana gösteriliyordu
# — bu iki sayının neden uyuşmadığı (personal item hariç tutulduğu için)
# arayüzde açık değildi, jüri/kullanıcı kafası karışabilirdi. Artık iki
# kategori AYRI ve ETİKETLİ gösteriliyor; doluluk oranının hangi sayıdan
# geldiği tek bakışta anlaşılıyor.
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Geçen Yolcu", f"{yolcu_no}/{ucus.toplam_yolcu if ucus else '—'}")
m2.metric(
    "🧳 Baş Üstü Dolabı",
    overhead_sayilan,
    help="Kabin bagajı + Oversized — overhead bin'i kullanan nesneler. Talep oranı bu sayıdan hesaplanır.",
)
m3.metric(
    "🎒 Koltuk Altı",
    personal_sayilan,
    help="Personal item (sırt çantası, laptop çantası vb.) — THY kuralına göre koltuk altına gider, overhead bin'i kullanmaz.",
)
m4.metric(
    "Baş Üstü Dolap Talebi",
    f"%{doluluk}",
    delta=f"+{doluluk:.1f}%" if doluluk > 0 else None,
    delta_color="inverse" if doluluk > 90 else "normal",
    help=f"{overhead_sayilan} (Baş Üstü Dolabı) / {kapasite} (Kapasite) — personal item dahil değil",
)
m5.metric("Kapasite", kapasite)

# ── AKSİYON BANNER ──
if cikti:
    sev   = cikti.aksiyon.seviye
    ikon  = {"normal":"✓","warning":"⚠","critical":"🚨"}.get(sev.value,"ℹ")
    klass = f"banner-{sev.value}"
    st.markdown(
        f'<div class="{klass}">'
        f'<strong>{ikon} AKSİYON:</strong> {cikti.aksiyon.mesaj}'
        f'<br/><small style="color:#8892a4">Tahmin: %{int(cikti.tahmini_doluluk*100)} | '
        f'Gerçek: %{int(cikti.gercek_doluluk*100)} | '
        f'Fark: {int((cikti.gercek_doluluk-cikti.tahmini_doluluk)*100):+d}%</small>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── GRAFİKLER ──
col_g1, col_g2 = st.columns([3, 2])

with col_g1:
    gecmis = list(st.session_state["doluluk_gecmis"].get(gate_id, []))
    if len(gecmis) > 1:
        xs = [g[0] for g in gecmis]
        ys = [g[1] for g in gecmis]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name="Gerçek talep oranı",
            line=dict(color="#e53e3e", width=2),
            fill="tozeroy",
            fillcolor="rgba(229,62,62,0.07)"
        ))
        if tahmin:
            t_pct = tahmin.tahmini_doluluk_orani * 100
            fig.add_hline(
                y=t_pct, line_dash="dash", line_color="#f6ad55",
                annotation_text=f"Tahmin %{int(t_pct)}",
                annotation_position="right",
            )
        fig.add_hline(
            y=90, line_dash="dot", line_color="#e53e3e",
            annotation_text="Kritik %90", annotation_position="right",
        )
        fig.add_hline(
            y=75, line_dash="dot", line_color="#f6ad55",
            annotation_text="Uyarı %75", annotation_position="right",
        )
        max_y = max(115, max(ys + ([tahmin.tahmini_doluluk_orani * 100] if tahmin else [])) * 1.15)
        fig.update_layout(
            title="Baş Üstü Dolap Talebi — Gerçek Zamanlı",
            xaxis_title="Yolcu Sırası",
            yaxis_title="Talep Oranı (%)",
            yaxis=dict(range=[0, max_y]),
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#8892a4", size=11),
            margin=dict(l=40, r=100, t=40, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor="#1c2330", zerolinecolor="#1c2330")
        fig.update_yaxes(gridcolor="#1c2330", zerolinecolor="#1c2330")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Boarding başlamadı. '▶ Başlat' butonuna basın.")

with col_g2:
    vals = [
        dagilim.get("oversized", 0),
        dagilim.get("cabin_ok", 0),
        dagilim.get("personal_item", 0),
    ]
    if sum(vals) > 0:
        fig2 = go.Figure(data=[go.Pie(
            labels=["Oversized", "Cabin OK", "Personal"],
            values=vals,
            hole=0.55,
            marker=dict(colors=["#e53e3e", "#48bb78", "#718096"]),
            textfont=dict(size=11),
        )])
        fig2.update_layout(
            title="Bagaj Dağılımı",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#8892a4", size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Etiketler
        st.markdown(
            f'<span class="tag tag-r">Oversized: {vals[0]}</span>'
            f'<span class="tag tag-g">Cabin OK: {vals[1]}</span>'
            f'<span class="tag tag-m">Personal: {vals[2]}</span>',
            unsafe_allow_html=True
        )

# ── ÖZGÜNLÜK 4: AKSİYON TIMELINE ──
seviye_gecmis = list(st.session_state["seviye_gecmis"].get(gate_id, []))
if len(seviye_gecmis) > 1:
    st.markdown("##### Aksiyon Zaman Çizelgesi")
    renk_map = {"normal": "#48bb78", "warning": "#f6ad55", "critical": "#e53e3e"}

    # Çok uzun geçmişi 60 segmente sıkıştır — performans + okunabilirlik
    if len(seviye_gecmis) > 60:
        step = len(seviye_gecmis) / 60
        ornekler = [seviye_gecmis[int(i*step)] for i in range(60)]
    else:
        ornekler = seviye_gecmis

    segs = "".join(
        f'<div class="timeline-seg" style="background:{renk_map.get(s,"#2d3748")}" '
        f'title="{s}"></div>'
        for s in ornekler
    )
    st.markdown(f'<div class="timeline-wrap">{segs}</div>', unsafe_allow_html=True)

    n_norm = seviye_gecmis.count("normal")
    n_warn = seviye_gecmis.count("warning")
    n_crit = seviye_gecmis.count("critical")
    ilk_warn_idx = next((i for i, s in enumerate(seviye_gecmis) if s == "warning"), None)
    ilk_crit_idx = next((i for i, s in enumerate(seviye_gecmis) if s == "critical"), None)

    st.markdown(
        f'<div class="timeline-legend">'
        f'<span><span class="timeline-dot" style="background:#48bb78"></span>Normal: {n_norm}</span>'
        f'<span><span class="timeline-dot" style="background:#f6ad55"></span>Dikkat: {n_warn}'
        + (f" (başladı: {ilk_warn_idx+1}. yolcu)" if ilk_warn_idx is not None else "")
        + f'</span>'
        f'<span><span class="timeline-dot" style="background:#e53e3e"></span>Kritik: {n_crit}'
        + (f" (başladı: {ilk_crit_idx+1}. yolcu)" if ilk_crit_idx is not None else "")
        + f'</span></div>',
        unsafe_allow_html=True
    )

# ── ÖZGÜNLÜK 3: TAHMİN İSABET ANALİZİ ──
if tahmin and len(gecmis_check := list(st.session_state["doluluk_gecmis"].get(gate_id, []))) > 5:
    with st.expander("🎯 Tahmin İsabet Analizi", expanded=False):
        final_gercek  = gecmis_check[-1][1]
        tahmin_pct    = tahmin.tahmini_doluluk_orani * 100
        sapma         = final_gercek - tahmin_pct
        sapma_abs     = abs(sapma)

        isabet_renk = "#48bb78" if sapma_abs < 10 else ("#f6ad55" if sapma_abs < 20 else "#e53e3e")
        isabet_metni = "Yüksek isabet" if sapma_abs < 10 else ("Orta isabet" if sapma_abs < 20 else "Düşük isabet")

        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Tahmin Edilen", f"%{tahmin_pct:.1f}")
        col_t2.metric("Şu An Gerçek", f"%{final_gercek:.1f}")
        col_t3.metric("Sapma", f"{sapma:+.1f}%", delta_color="off")

        st.markdown(
            f'<div style="text-align:center;padding:6px;background:rgba(0,0,0,0.2);'
            f'border-radius:6px;color:{isabet_renk};font-weight:600;font-size:13px">'
            f'{isabet_metni} — Tahmin güveni %{int(tahmin.guven_skoru*100)} '
            f'({tahmin.tahmin_metodu})</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Sistem her boarding sonunda gerçek sonucu sefer hafızasına kaydeder — "
            "bir sonraki aynı hat tahmini bu veriyle güçlenir."
        )

# ── ZAMAN DAMGASI ──
st.caption(
    f"Gate: {gate_id} | "
    f"Threshold: {inf.aktif_threshold if inf else '—'} | "
    f"{time.strftime('%H:%M:%S')}"
)

# ── OTOMATİK YENİLEME ──
if st.session_state["sim_calisıyor"]:
    time.sleep(bekleme)
    st.rerun()
