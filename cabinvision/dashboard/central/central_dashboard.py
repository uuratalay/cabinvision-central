# dashboard/central/central_dashboard.py
#
# Operasyon merkezinin ekranı.
# Tüm terminal'in gate durumları, aktif uyarılar, sefer geçmişi.
# Streamlit'te çalıştır: streamlit run dashboard/central/central_dashboard.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from dashboard.state_manager import (
    get_or_init_system, sim_adim, yeni_ucus_baslat
)
from central.models.flight_models import DolulukSeviyesi

st.set_page_config(
    page_title="CabinVision | Operations Center",
    page_icon="🏢",
    layout="wide",
)

st.markdown("""
<style>
*{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.stApp{background:#0a0d13;color:#f0f4f8}
.block-container{padding-top:1.5rem}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #2d3748}
[data-testid="metric-container"]{
    background:#161b22;border:1px solid #2d3748;
    border-radius:8px;padding:12px;
}

/* Gate durum kartları */
.gate-card{
    background:#161b22;border:1px solid #2d3748;
    border-radius:10px;padding:16px;margin-bottom:10px;
    transition:border-color .3s;
}
.gate-card.normal{border-color:#2d3748}
.gate-card.warning{border-color:#f6ad55;background:rgba(246,173,85,0.05)}
.gate-card.critical{
    border-color:#e53e3e;background:rgba(229,62,62,0.07);
    animation:pcrit 1.2s infinite;
}
@keyframes pcrit{0%,100%{box-shadow:none}50%{box-shadow:0 0 12px rgba(229,62,62,0.12)}}

.badge{
    display:inline-block;padding:3px 10px;border-radius:12px;
    font-size:11px;font-weight:700;
}
.badge-n{background:rgba(72,187,120,0.15);color:#48bb78}
.badge-w{background:rgba(246,173,85,0.15);color:#f6ad55}
.badge-c{background:rgba(229,62,62,0.15);color:#e53e3e}

.pbar-bg{background:#1c2330;border-radius:4px;height:8px;margin:6px 0}
.pbar-fill{height:8px;border-radius:4px}
</style>
""", unsafe_allow_html=True)

# ── SİSTEM BAŞLAT ──
GATE_IDS = [
    "IST-GATE-07", "IST-GATE-12", "IST-GATE-18",
    "IST-GATE-23", "IST-GATE-31", "IST-GATE-42",
]
get_or_init_system(GATE_IDS)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🏢 CabinVision")
    st.markdown("**Operations Center**")
    st.divider()

    sim_hizi = st.select_slider(
        "Global Simülasyon Hızı",
        options=["Yavaş", "Normal", "Hızlı"],
        value="Normal",
    )
    hiz_map = {"Yavaş": 0.4, "Normal": 0.15, "Hızlı": 0.03}
    bekleme = hiz_map[sim_hizi]

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Tümünü Başlat", use_container_width=True, type="primary"):
            st.session_state["sim_calisıyor"] = True
    with col2:
        if st.button("⏸ Durdur", use_container_width=True):
            st.session_state["sim_calisıyor"] = False

    if st.button("🔄 Tümü Yeni Uçuş", use_container_width=True):
        for gid in GATE_IDS:
            yeni_ucus_baslat(gid)
        st.session_state["sim_calisıyor"] = False
        st.rerun()

    st.divider()

    # Audit log özeti
    audit = st.session_state.get("audit_obs")
    if audit:
        st.caption(f"Audit log: {audit.log_boyutu} kayıt")

    # Memory istatistiği
    mem = st.session_state.get("memory_repo")
    if mem:
        st.caption(f"Sefer hafızası: {mem.toplam_kayit_sayisi} uçuş")

# ── BAŞLIK ──
st.markdown("## Operasyon Merkezi — Terminal İstanbul")
st.caption(f"Son güncelleme: {time.strftime('%H:%M:%S')} | {len(GATE_IDS)} aktif gate")
st.divider()

# ── BOARDING ADIMI (tüm gate'ler) ──
if st.session_state["sim_calisıyor"]:
    for gid in GATE_IDS:
        sim_adim(gid)

# ── ÖZET METRİKLER ──
central_obs = st.session_state.get("central_obs")
aktif_uyarilar = central_obs.aktif_uyarilar if central_obs else []
kritik_sayisi  = sum(
    1 for c in aktif_uyarilar
    if c.aksiyon.seviye == DolulukSeviyesi.CRITICAL
)
uyari_sayisi = sum(
    1 for c in aktif_uyarilar
    if c.aksiyon.seviye == DolulukSeviyesi.WARNING
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Aktif Gate", len(GATE_IDS))
m2.metric("Kritik Uyarı", kritik_sayisi, delta=f"+{kritik_sayisi}" if kritik_sayisi > 0 else None, delta_color="inverse")
m3.metric("Dikkat", uyari_sayisi)
m4.metric("Normal", len(GATE_IDS) - len(aktif_uyarilar))

st.divider()

# ── GATE DURUM KARTLARI ──
st.markdown("### Gate Durumları")

cols = st.columns(3)
for i, gate_id in enumerate(GATE_IDS):
    inf    = st.session_state["gate_services"].get(gate_id)
    ucus   = st.session_state["gate_ucuslar"].get(gate_id)
    tahmin = st.session_state["gate_tahmins"].get(gate_id)
    fusion = st.session_state["gate_engines"].get(gate_id)
    yolcu  = st.session_state["sim_yolcu_no"].get(gate_id, 0)

    # GEMINI 3. TUR MADDE 3 (B): doluluk hesabı personal item hariç
    overhead_sayilan = inf.overhead_bin_sayilan if inf else 0
    kapasite = ucus.dolap_kapasitesi if ucus else 120
    doluluk  = round(overhead_sayilan / kapasite * 100, 1)
    dagilim  = inf.boyut_dagilimi if inf else {}

    # Mevcut durum
    son_cikti = None
    if fusion and ucus:
        son_cikti = fusion.son_cikti(gate_id, ucus.ucus_no)

    sev_val = "normal"
    sev_mesaj = "Normal — Boarding devam ediyor"
    if son_cikti:
        sev_val   = son_cikti.aksiyon.seviye.value
        sev_mesaj = son_cikti.aksiyon.mesaj[:70]

    badge_class = {"normal": "badge-n", "warning": "badge-w", "critical": "badge-c"}[sev_val]
    badge_text  = {"normal": "NORMAL", "warning": "DİKKAT", "critical": "KRİTİK"}[sev_val]

    # Kapasite bar rengi
    bar_color = "#e53e3e" if doluluk >= 90 else ("#f6ad55" if doluluk >= 75 else "#48bb78")

    with cols[i % 3]:
        st.markdown(f"""
        <div class="gate-card {sev_val}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <strong style="font-size:14px">{gate_id}</strong>
                <span class="badge {badge_class}">{badge_text}</span>
            </div>
            <div style="font-size:12px;color:#8892a4;margin-bottom:4px">
                {ucus.ucus_no if ucus else '—'} &nbsp;|&nbsp; {ucus.hat if ucus else '—'}
            </div>
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:8px">
                <span>Talep: <strong style="color:{bar_color}">%{doluluk}</strong></span>
                <span style="color:#8892a4">{yolcu}/{ucus.toplam_yolcu if ucus else '—'} yolcu</span>
            </div>
            <div class="pbar-bg">
                <div class="pbar-fill" style="width:{min(doluluk,100)}%;background:{bar_color}"></div>
            </div>
            <div style="font-size:11px;color:#8892a4;margin-top:6px" title="Oversized + Cabin OK = Baş Üstü Dolabı (talebe dahil) | Personal = Koltuk Altı (talebe dahil DEĞİL)">
                <span style="color:#e53e3e">▲ {dagilim.get('oversized',0)} Oversize</span> &nbsp;
                <span style="color:#48bb78">▲ {dagilim.get('cabin_ok',0)} Kabin</span> &nbsp;
                <span style="color:#718096">▲ {dagilim.get('personal_item',0)} Koltuk Altı*</span>
            </div>
            <div style="font-size:9px;color:#5a6472;margin-top:2px">* Koltuk altı talebe dahil değil</div>
            {f'<div style="font-size:11px;color:#8892a4;margin-top:6px;border-top:1px solid #2d3748;padding-top:6px">{sev_mesaj}</div>' if sev_val != "normal" else ""}
        </div>
        """, unsafe_allow_html=True)

# ── TERMİNAL DOLULUK GRAFİĞİ ──
st.divider()
st.markdown("### Terminal Baş Üstü Talep Genel Bakış")
st.caption(
    "Baş Üstü Dolabı (overhead bin doluluğunu belirler) ile Koltuk Altı "
    "(personal item, talebe dahil değil) ayrı renklerle gösterilir."
)

# GEMINI 5. TUR MADDE 1 DÜZELTMESİ (kabul edildi):
# Gate seviyesinde (mikro) çözülen Baş Üstü/Koltuk Altı netliği, terminal
# seviyesinde (makro) de aynı şekilde gösterilmeli — aksi halde tutarsızlık
# sadece tek-gate ekranında çözülmüş, çoklu-gate genel bakışta hâlâ var olur.
gate_names, bas_ustu_sayilar, koltuk_alti_sayilar, doluluklar, renkler = [], [], [], [], []
for gid in GATE_IDS:
    inf      = st.session_state["gate_services"].get(gid)
    ucus     = st.session_state["gate_ucuslar"].get(gid)
    dagilim  = inf.boyut_dagilimi if inf else {}

    overhead_sayilan = inf.overhead_bin_sayilan if inf else 0
    personal_sayilan = dagilim.get("personal_item", 0)
    kapasite = ucus.dolap_kapasitesi if ucus else 120
    d        = round(overhead_sayilan / kapasite * 100, 1)

    gate_names.append(gid.split("-")[-1])
    bas_ustu_sayilar.append(overhead_sayilan)
    koltuk_alti_sayilar.append(personal_sayilan)
    doluluklar.append(d)
    renkler.append(
        "#e53e3e" if d >= 90 else ("#f6ad55" if d >= 75 else "#48bb78")
    )

fig = go.Figure()
fig.add_trace(go.Bar(
    x=gate_names, y=bas_ustu_sayilar,
    name="Baş Üstü Dolabı (talebe dahil)",
    marker_color=renkler,
    text=[f"%{d}" for d in doluluklar],
    textposition="outside",
))
fig.add_trace(go.Bar(
    x=gate_names, y=koltuk_alti_sayilar,
    name="Koltuk Altı (talebe dahil DEĞİL)",
    marker_color="#718096",
    opacity=0.55,
))
fig.update_layout(
    barmode="group",
    title="Gate Bazlı Bagaj Dağılımı — Baş Üstü vs Koltuk Altı",
    yaxis_title="Bagaj sayısı",
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#8892a4", size=11),
    margin=dict(l=40, r=40, t=40, b=40),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
)
fig.update_xaxes(gridcolor="#1c2330")
fig.update_yaxes(gridcolor="#1c2330")
st.plotly_chart(fig, use_container_width=True)

# Ayrı, sade bir doluluk-yüzdesi grafiği (eski grafiğin sadeleştirilmiş hali)
fig2 = go.Figure(go.Bar(
    x=gate_names, y=doluluklar, marker_color=renkler,
    text=[f"%{d}" for d in doluluklar], textposition="auto",
))
fig2.add_hline(y=90, line_dash="dot", line_color="#e53e3e", annotation_text="Kritik %90")
fig2.add_hline(y=75, line_dash="dot", line_color="#f6ad55", annotation_text="Uyarı %75")
fig2.update_layout(
    title="Gate Bazlı Baş Üstü Dolap Talep Oranı (%)",
    yaxis=dict(range=[0, max(115, max(doluluklar or [0]) * 1.15)]),
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#8892a4", size=11),
    margin=dict(l=40, r=40, t=40, b=40),
    showlegend=False,
)
fig2.update_xaxes(gridcolor="#1c2330")
fig2.update_yaxes(gridcolor="#1c2330")
st.plotly_chart(fig2, use_container_width=True)

# ── YÜKSEK RİSKLİ HATLAR ──
col_r1, col_r2 = st.columns(2)

with col_r1:
    st.markdown("### 🧠 Sefer Hafızası — Yüksek Riskli Hatlar")
    mem = st.session_state.get("memory_repo")
    if mem:
        riskli = mem.yuksek_riskli_hatlar(doluluk_esigi=0.70)
        if riskli:
            st.caption(
                f"{mem.toplam_kayit_sayisi} geçmiş uçuştan öğrenilen pattern'ler — "
                "bu hatlar tahmin motorunu otomatik olarak güçlendiriyor."
            )
            for ist in riskli[:5]:
                pct = int(ist.ort_doluluk * 100)
                over_pct = int(ist.ort_oversized_oran * 100)
                renk = "#e53e3e" if pct >= 90 else "#f6ad55"
                st.markdown(
                    f"**{ist.hat}** &nbsp; "
                    f"<span style='color:{renk}'>Ort. %{pct} talep</span> &nbsp; "
                    f"<span style='color:#a78bfa'>Oversized: %{over_pct}</span> &nbsp; "
                    f"<span style='color:#8892a4;font-size:12px'>{ist.kayit_sayisi} uçuş · güven %{int(ist.guven_skoru*100)}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Yeterli veri yok.")

with col_r2:
    st.markdown("### Son Audit Kayıtları")
    audit = st.session_state.get("audit_obs")
    if audit and audit.log_boyutu > 0:
        for kayit in reversed(audit.son_n_kayit(6)):
            sev   = kayit["seviye"]
            renk  = {"normal":"#48bb78","warning":"#f6ad55","critical":"#e53e3e"}.get(sev,"#8892a4")
            st.markdown(
                f"<span style='color:{renk};font-size:11px'>●</span> "
                f"<span style='font-size:12px'>{kayit['gate_id'].split('-')[-1]} | "
                f"{kayit['ucus_no']} | "
                f"%{int(kayit['gercek_doluluk']*100)} talep</span>",
                unsafe_allow_html=True
            )
    else:
        st.info("Henüz log yok.")

# ── OTOMATİK YENİLEME ──
if st.session_state["sim_calisıyor"]:
    time.sleep(bekleme)
    st.rerun()
