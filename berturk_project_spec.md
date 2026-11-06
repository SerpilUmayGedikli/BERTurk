# Proje Şartnamesi: Türkçe Metin Sınıflandırma (BERTurk İnce Ayarı)

> Bu dosya bir kodlama ajanına (Claude Code vb.) verilmek üzere yazıldı.
> Ajan için kural: önce bu dosyanın tamamını oku, sonra sadece istenen aşamayı yap.
> Link ve sürüm bilgilerini kodlamadan önce güncel dokümantasyondan doğrula.

## 1. Amaç

Türkçe metinler için iki görevde ince ayarlı bir BERT modeli geliştirmek ve klasik bir baseline ile **ölçülebilir şekilde** kıyaslamak:

- **Görev A:** Haber kategorisi sınıflandırma (çok sınıflı)
- **Görev B:** Duygu analizi (olumlu / olumsuz / nötr, veri setinin etiketlerine göre)

Teslimatlar: çalışan eğitim kodu, tekrarlanabilir deneyler, kıyas tablosu, hata analizi, canlı demo (Hugging Face Spaces), README.

Bu proje bir CV portföy projesidir. Bu yüzden **ölçüm dürüstlüğü** her şeyden önemlidir (bkz. Bölüm 10).

## 2. Ücretsiz kaynak ve anahtar politikası

Projedeki tüm bileşenler ücretsiz olmalı. Ücretli servis, kredi kartı gerektiren servis veya ücretli API **kullanılmayacak**.

| İhtiyaç | Araç | Anahtar gerekir mi? |
|---|---|---|
| Model ve veri | Hugging Face Hub (herkese açık model ve veri setleri) | İndirmek için hayır |
| Modeli Hub'a yükleme (opsiyonel) | Hugging Face hesabı + ücretsiz Access Token (write) | Evet, ücretsiz |
| Eğitim GPU'su | Google Colab (ücretsiz T4) veya Kaggle Notebooks (ücretsiz GPU, haftalık kota) | Hayır |
| Deney takibi | Yerel CSV/JSON log + TensorBoard (varsayılan). Opsiyonel: MLflow (yerel) | Hayır |
| Demo | Gradio + Hugging Face Spaces (ücretsiz CPU Space) | HF hesabı yeterli |
| CI | GitHub Actions (herkese açık repo için ücretsiz) | Hayır |

Kural: Secret'lar yalnızca ortam değişkeni veya Colab/Kaggle Secrets ile okunur. Anahtar koda, not defterine, commit'e veya log'a **yazılmaz**. `.env` dosyası `.gitignore` içindedir.

Gerekli tek anahtar: `HF_TOKEN` (ücretsiz, sadece Hub'a yükleme için, opsiyonel).

## 3. Dokümantasyon kaynakları (hepsi ücretsiz)

Ajan bu sayfaları okuyarak çalışsın. Sürüm farkı olursa güncel dokümantasyon önceliklidir.

| Konu | Adres |
|---|---|
| Transformers: sequence classification rehberi | https://huggingface.co/docs/transformers/tasks/sequence_classification |
| Transformers: Trainer API | https://huggingface.co/docs/transformers/main_classes/trainer |
| Transformers: Auto sınıfları | https://huggingface.co/docs/transformers/model_doc/auto |
| Datasets kütüphanesi | https://huggingface.co/docs/datasets |
| Evaluate kütüphanesi | https://huggingface.co/docs/evaluate |
| Hugging Face Hub: model yükleme | https://huggingface.co/docs/hub/models-uploading |
| Hugging Face Spaces (Gradio) | https://huggingface.co/docs/hub/spaces-sdks-gradio |
| Gradio dokümantasyonu | https://www.gradio.app/docs |
| scikit-learn: TF-IDF ve LogisticRegression | https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting |
| scikit-learn: metrikler (f1, confusion matrix) | https://scikit-learn.org/stable/modules/model_evaluation.html |
| PyTorch dokümantasyonu | https://pytorch.org/docs/stable |
| Colab SSS | https://research.google.com/colaboratory/faq.html |
| Kaggle Notebooks GPU | https://www.kaggle.com/docs/notebooks |
| uv (paket yönetimi) | https://docs.astral.sh/uv |
| pytest | https://docs.pytest.org |
| Ruff | https://docs.astral.sh/ruff |

Model kartı: `dbmdz/bert-base-turkish-cased` (BERTurk). Hub'da model sayfasındaki lisans ve kullanım notlarını README'ye aktar.

## 4. Veri setleri

Veri setlerini Hugging Face Hub'dan `datasets.load_dataset` ile yükle. **Kullanmadan önce** veri kartını oku; lisans, etiketler ve bölünme (split) yapısını doğrula. Burada adı geçen setler başlangıç adaylarıdır, erişilemezse Hub'da "turkish news classification" ve "turkish sentiment" aramasıyla alternatif bul ve seçimini README'de gerekçelendir.

| Görev | Aday veri seti (Hub kimliği) | Doğrulanacaklar |
|---|---|---|
| A: Haber kategorisi | `savasy/ttc4900` | Sınıf sayısı ve dağılımı, train/test bölünmesi var mı |
| B: Duygu | `winvoker/turkish-sentiment-analysis-dataset` | Etiket sayısı, sınıf dengesi, lisans |

Veri kuralları:
- Test setine eğitim sırasında **hiç bakma**. Validation yoksa train'den sabit seed ile ayır (%10-15, stratified).
- Aynı metin hem train hem test'te var mı kontrol et (sızıntı/duplicate). Varsa raporla ve temizle.
- Sınıf dengesizliği varsa raporla; metrik olarak accuracy yerine **macro-F1** ana metrik olsun.
- Ham veriyi repoya commit etme. Veri `data/` altına indirilir, `.gitignore` içindedir.

## 5. Repo yapısı

```
turkish-text-classification/
├── pyproject.toml
├── README.md
├── configs/
│   ├── news_berturk.yaml
│   └── sentiment_berturk.yaml
├── src/tcls/
│   ├── __init__.py
│   ├── data.py            # yükleme, temizleme, split, tokenization
│   ├── baseline.py        # TF-IDF + LogisticRegression
│   ├── train.py           # BERTurk ince ayarı (Trainer)
│   ├── evaluate.py        # metrikler, confusion matrix, hata dökümü
│   ├── predict.py         # tek metin tahmini
│   └── utils.py           # seed, logging, config okuma
├── notebooks/
│   └── colab_train.ipynb  # Colab/Kaggle'da çalıştırma kabuğu
├── app/
│   ├── app.py             # Gradio demo
│   └── requirements.txt   # Space için
├── tests/
├── reports/               # metrik tabloları, grafikler, hata analizi
└── .github/workflows/ci.yml
```

## 6. Bağımlılıklar

Python 3.11. Paket yönetimi `uv`. Başlangıç seti (sürümleri `uv add` ile güncel al):

`torch`, `transformers`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `gradio`; geliştirme için: `pytest`, `ruff`, `mypy`.

## 7. Aşamalar ve kabul kriterleri

Her aşama ayrı commit. Bir aşamanın kabul kriterleri sağlanmadan sonrakine geçme.

### Aşama 1: İskelet ve veri
- Repo yapısı, `pyproject.toml`, ruff/mypy/pytest yapılandırması, CI.
- `data.py`: yükleme, temizleme (boş metin, duplicate), stratified split, etiket eşlemesi.
- Kabul: veri özeti (örnek sayısı, sınıf dağılımı, ortalama/p95 token uzunluğu) `reports/data_summary.md` içine yazılır; testler (en az 6) geçer; CI yeşil.

### Aşama 2: Baseline
- `baseline.py`: TF-IDF (kelime 1-2 gram) + LogisticRegression. Basit bir ikinci baseline (ör. LinearSVC) opsiyonel.
- Kabul: macro-F1, accuracy, sınıf bazlı precision/recall/F1 `reports/baseline_*.json` dosyasına yazılır; sonuçlar 3 farklı seed ile (varsa) ortalama ± std olarak raporlanır.

### Aşama 3: BERTurk ince ayarı
- `train.py`: `AutoModelForSequenceClassification` + `Trainer`, config'ten okunan hiperparametreler.
- Başlangıç: max_length 128 (veri uzunluk dağılımına göre 256'ya çıkar), lr 2e-5, batch 16-32, 3-4 epoch, weight decay 0.01, warmup oranı 0.1, fp16 (GPU varsa), erken durdurma (validation macro-F1).
- Ücretsiz GPU sınırı: Colab T4'te bellek yetmezse batch'i düşür, `gradient_accumulation_steps` kullan.
- Kabul: 3 farklı seed ile eğitim, test macro-F1 ortalama ± std; checkpoint'ler repoya değil Drive/Kaggle çıktısına gider.

### Aşama 4: Değerlendirme ve ablation
- Confusion matrix grafiği, sınıf bazlı rapor.
- En az 2 ablation: (a) `max_length` 128 vs 256, (b) öğrenme oranı 2e-5 vs 5e-5 (ya da cased vs uncased model karşılaştırması).
- **Hata analizi** (`reports/error_analysis.md`): en çok karışan sınıf çiftleri, en az 20 yanlış tahmin örneği, her biri için olası neden (kısa metin, belirsiz etiket, etiket gürültüsü, alan kayması vb.). Nedenler gözleme dayanmalı, tahmin olarak işaretlenmeli.
- Kabul: kıyas tablosu (baseline vs BERTurk) `reports/results.md` içinde.

### Aşama 5: Demo
- Gradio uygulaması: metin kutusu, tahmin edilen sınıf ve güven skorları (softmax), 3 örnek metin.
- Model Hub'a yüklenmişse oradan, yüklenmediyse Space içine yerel dosya olarak yüklenir. CPU Space'te çalışacak boyutta olmalı.
- Kabul: Space linki README'de; ilk tahmin birkaç saniyede döner.

### Aşama 6: Cilalama
- README: problem, veri, yöntem, sonuç tablosu, hata analizi özeti, sınırlılıklar, tekrar üretme komutları, lisans notları.
- Model kartı (`MODEL_CARD.md`): veri, amaçlanan kullanım, bilinen sınırlar.

## 8. Ücretsiz GPU'da çalıştırma notları

- Colab: Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU. Oturum süresi ve kota sınırlıdır, checkpoint'i düzenli kaydet (`save_strategy="epoch"`, `save_total_limit=2`).
- Kaggle: Notebook ayarlarında Accelerator'ı GPU seç; haftalık GPU kotasını Kaggle hesap sayfasından takip et.
- `notebooks/colab_train.ipynb` yalnızca repoyu klonlayıp `python -m tcls.train --config ...` çalıştırmalıdır. Mantık notebook'ta değil `src/` içinde olmalı.
- Eğitim kodu CPU'da küçük bir alt küme (örn. 200 örnek, 1 epoch) ile **smoke test** olarak çalışabilmelidir (`--smoke`).

## 9. Kod kalitesi kuralları

- Type hint her yerde, `ruff check`, `ruff format --check`, `mypy`, `pytest` yeşil olmadan aşama bitmez.
- Tüm rastgelelik tek bir `set_seed` ile kontrol edilir.
- Testlerde gerçek model indirilmez; tokenizer/model mock'lanır veya çok küçük sahte veri kullanılır. Gerçek model gerektiren test `slow` olarak işaretlenir ve varsayılan koşuda çalışmaz.
- Config dosyaları dışında sabit hiperparametre yok.
- Secret loglama yok.

## 10. Dürüstlük kuralları (kritik)

- Çalıştırılmamış hiçbir sonucu rapora, README'ye veya CV maddesine **yazma**. Sayıları yalnızca gerçekten çalıştırdığın deneylerden al.
- Çalıştıramadığın şeyi (GPU yok, kota bitti, ağ yok) açıkça "ÇALIŞTIRILMADI" olarak işaretle.
- Test setiyle hiperparametre seçme. Seçim validation ile yapılır, test yalnızca bir kez, en sonda kullanılır.
- Sonuç tablosunda baseline'ı da mutlaka göster. BERTurk baseline'dan iyi çıkmazsa bunu olduğu gibi raporla ve nedenini araştır.

## 11. Ajana verilecek başlangıç prompt'u

```
Read berturk_project_spec.md fully. Do ONLY "Aşama 1" now (repo skeleton,
data loading/cleaning/splitting, tests, CI). Follow the free-resources policy
and the honesty rules. Verify dataset names and library APIs against the docs
linked in the spec before coding. Run ruff, mypy and pytest, report exact
results, commit, and stop. Do not start Aşama 2.
```

Sonraki aşamalar için aynı prompt'ta yalnızca aşama adını değiştir.

## 12. CV maddesi şablonu (gerçek rakamlarla doldur)

- Fine-tuned **BERTurk** for Turkish news categorization (N classes) and sentiment analysis, reaching macro-F1 of **X** (vs **Y** for a TF-IDF + logistic regression baseline), averaged over 3 seeds.
- Ran ablations on sequence length and learning rate, and documented an error analysis of the most confused classes.
- Deployed an interactive demo on **Hugging Face Spaces** using Gradio, with CI (ruff, mypy, pytest) on GitHub Actions.
