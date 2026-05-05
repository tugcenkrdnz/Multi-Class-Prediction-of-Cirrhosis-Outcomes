
# Kaggle S3E26 - 🩺 Siroz Hastalığı Sağkalım Tahmin Sistemi

Bu proje, karaciğer sirozu olan hastaların klinik ve laboratuvar parametrelerini analiz ederek, hastanın gelecekteki durumunu (Yaşıyor, Nakil, Vefat) tahmin eden bir makine öğrenmesi uygulamasıdır.

## 🚀 Proje Hakkında

Proje, tıbbi verilerdeki karmaşık ilişkileri ve sınıf dengesizliğini yönetmek amacıyla oluşturulmuştur. Özellikle veri setinde azınlıkta olan "Nakil (CL)" vakalarını tespit edebilmek için özel ağırlıklandırma teknikleri kullanılmıştır.

### Temel Özellikler
*   **Hibrit Model (Blending):** CatBoost, XGBoost ve LightGBM modellerinin ağırlıklı ortalaması alınarak tek bir güçlü tahminci oluşturulmuştur.
*   **Özellik Mühendisliği:** Ham veriler üzerinden logaritmik dönüşümler, oran analizleri ve enzim indeksleri türetilmiştir.
*   **Olasılık Kalibrasyonu:** LogLoss değerini optimize etmek için tahmin olasılıkları kalibre edilmiştir.
*   **İnteraktif Arayüz:** Streamlit kütüphanesi kullanılarak doktorların veya araştırmacıların kullanımı için kullanıcı dostu bir panel tasarlanmıştır.

## 📊 Veri Seti Detayları

Model eğitilirken aşağıdaki kritik parametreler kullanılmıştır:
- **Klinik:** Takip gün sayısı, yaş, ödem durumu, asit varlığı, karaciğer büyümesi.
- **Laboratuvar:** Bilirubin, Albumin, Bakır, Kolesterol, SGOT (AST), Alk Phos, Trigliserit, Trombosit, Prothrombin süresi.

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler
Projenin çalışması için bilgisayarınızda Python yüklü olmalıdır. Gerekli kütüphaneleri yüklemek için:

```bash
pip install streamlit pandas numpy joblib xgboost lightgbm catboost scikit-learn
```

### Uygulamayı Başlatma
1. Proje dosyalarını indirin.
2. Terminal veya komut istemcisini açarak proje klasörüne gidin.
3. Aşağıdaki komutu çalıştırın:

```bash
streamlit run app.py
```

## 🧠 Model Performansı

Yapılan deneyler sonucunda elde edilen en iyi değerler:
*   **Eğitim LogLoss:** ~ 0.22
*   **Public Score (Kaggle):** ~0.42

## 📁 Dosya Yapısı

*   `app.py`: Streamlit arayüz kodlarını içeren ana dosya.
*   `siroz_model_final.pkl`: Eğitilmiş modelleri, özellik sıralamasını ve encoder nesnelerini içeren binary dosya.
*   `notebook.ipynb`: Veri analizi, görselleştirme ve eğitim süreçlerinin yürütüldüğü çalışma dosyası.

## ⚠️ Önemli Not
Bu uygulama bir **karar destek mekanizmasıdır.** Tahmin sonuçları tıbbi bir tavsiye niteliği taşımaz; kesin sonuçlar için uzman bir hekim onayı gereklidir.
