# CabinVision v11 — Professional Engineering Fixes

Bu sürüm, `cabinvision_full_v10.zip` üzerinde yapılan profesyonel yazılım mühendisliği düzenlemelerini içerir.
Amaç yalnızca testleri geçirmek değil; dış denetimde yakalanabilecek test altyapısı sorunlarını, gizli varsayımları ve operasyonel dayanıklılık eksiklerini daha açık, sürdürülebilir ve test edilebilir hale getirmektir.

## 1. Test altyapısı düzeltildi

### Problem
`pytest -q` çalıştırıldığında `tests/test_integration.py` içindeki bazı testler hata veriyordu. Sebep, test fonksiyonlarının birbirine argüman geçirmesi; pytest'in bu argümanları fixture sanmasıydı.

### Çözüm
- `tests/test_integration.py` standart pytest fixture yapısına geçirildi.
- `calibration`, `inference_service`, `memory_repo`, `prediction_pair` fixture'ları eklendi.
- Eski manuel runner korunmadı; `python tests/test_integration.py` artık içeriden pytest çağırır.
- Pytest warning üreten `return result` kullanımları kaldırıldı.

### Sonuç
`python -m pytest -q` sonucu:

```text
28 passed in 1.42s
```

## 2. AdaptiveThreshold davranışı profesyonelleştirildi

### Problem
Eski yapı şu formülle threshold'u gizlice değiştiriyordu:

```python
base + (mean_conf - 0.65) * 0.15
```

`0.65` ve `0.15` veriyle doğrulanmamıştı. Model zorlandığında threshold'un düşmesi yanlış pozitifleri artırabilirdi.

### Çözüm
- `AdaptiveThreshold` dış API uyumluluğu için korundu.
- Eşik artık sabit kalır.
- Son frame confidence istatistiği yalnızca `quality_flag` üretmek için kullanılır.
- `InferenceService.scene_quality_flag` eklendi.

### Sonuç
Detector threshold'u artık gizli şekilde değişmez; görüntü/model güveni ayrı raporlanır.

## 3. PredictionEngine varsayımları şeffaflaştırıldı

### Problem
Eski tahmin motoru şu kanıtsız varsayıma dayanıyordu:

```python
beyan_etmeyen * 0.40
```

Ayrıca geçmiş veri ağırlığı `0.35` tavanıyla sabitlenmişti.

### Çözüm
- `%40 beyan etmeyen getirir` varsayımı kaldırıldı.
- Tahminin ana gövdesi sefer hafızası + uçak kapasitesi oldu.
- Check-in/PNR sinyali doğrulanmadığı için yalnızca düşük ağırlıklı opsiyonel feature olarak kullanılıyor.
- Geçmiş veri ağırlığı kayıt sayısına göre açıklanabilir hale getirildi:

```python
gecmis_agirlik = min(0.70, (kayit_sayisi / 50) * 0.70)
```

- Tahmini toplam bagaj sayısı hâlâ yolcu sayısıyla clamp ediliyor.

### Sonuç
PredictionEngine artık kanıtsız tek sabitle değil, daha okunabilir ve kalibre edilebilir bir harmanlama mantığıyla çalışır.

## 4. FusionEngine karar mantığı iyileştirildi

### Problem
FusionEngine yalnızca anlık gerçek oran ve sabit eşiklerle karar veriyordu.

### Çözüm
- Tahmini ham bagaj sayısı kapasiteye bölünerek karar motoruyla aynı birime taşındı.
- Karar oranı, mevcut talep ve tahmini toplam talebin maksimumu olarak değerlendiriliyor.
- `SistemGuveni` eklendi (`YUKSEK`, `ORTA`, `DUSUK`).
- Alert seviyesinin bir anda aşağı zıplamasını önlemek için monotonic escalation mantığı eklendi.

### Sonuç
Dashboard tarafında sistem artık hem risk seviyesini hem de tahmin güvenini taşıyabilir.

## 5. EventPublisher dayanıklılığı artırıldı

### Problem
Outbox veri kaybını azaltıyordu; ancak ağ dalgalanmasında retry fırtınası ve tekrar işleme riski vardı.

### Çözüm
- Basit circuit breaker eklendi.
- Event payload'larına `event_id` eklendi.
- Lokal idempotency koruması eklendi.
- Circuit açıkken event'ler kaybedilmez, sıraya geri konur.

### Sonuç
Edge node, ağ hatasında merkezi sistemi gereksiz retry yüküyle zorlamaz.

## 6. Doğrulama sonuçları

```text
python -m pytest -q
28 passed

python tests/test_integration.py
7 passed

python tests/edge/test_calibration_service.py
9 passed, 0 failed

python tests/test_c1_c2_fixes.py
12 passed, 0 failed
```

## 7. Bilinçli olarak açık kalanlar

Bu sürüm teknik borcu azaltır fakat gerçek operasyonel doğruluğu kanıtlamaz. Aşağıdaki maddeler hâlâ pilot veri/ürünleşme aşamasında ele alınmalıdır:

- Personal item getirme oranının gerçek veriyle kalibrasyonu.
- Narrow/wide kapasite modelinin uçak tipi/kabin konfigürasyonu seviyesine indirilmesi.
- Boarding progress ve gerçek yolcu geçiş sayısının daha güçlü bir Time-to-Full modeline bağlanması.
- Bagaj-yolcu ilişkilendirmesi.
- KVKK/hukuki veri yönetimi katmanı.

## Mühendislik notu

Bu sürümün hedefi "sistem artık gerçek hayatta kesin doğru tahmin yapar" iddiası değildir. Doğru iddia şudur:

> Sistem artık standart pytest ile doğrulanabilir, daha şeffaf varsayımlar kullanan, fiziksel/olasılıksal tutarlılığı güçlendirilmiş ve dış denetime daha dayanıklı bir prototiptir.
