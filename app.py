import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Sayfa Yapılandırması
st.set_page_config(page_title="Siroz Sağkalım Tahmini", layout="wide")
st.title("🩺 Siroz Hastalığı Tahmin Sistemi")
st.write("Lütfen hastanın klinik ve laboratuvar verilerini giriniz.")

# 2. Model ve Gerekli Verilerin Yüklenmesi
@st.cache_resource
def load_model_assets():
    # Bu dosyanın Notebook'ta kaydettiğiniz isimle aynı olduğundan emin olun
    return joblib.load('siroz_model_final.pkl')

try:
    assets = load_model_assets()
except Exception as e:
    st.error(f"Model dosyası yüklenemedi: {e}")
    st.stop()

# 3. Kullanıcı Giriş Paneli (Sidebar)
st.sidebar.header("📋 Hasta Klinik Verileri")

def get_user_inputs():
    # Sayısal Değerler
    n_days = st.sidebar.number_input("Takip Gün Sayısı (N_Days)", 0, 5000, 1500)
    age = st.sidebar.number_input("Yaş (Gün)", 0, 30000, 18000)
    bilirubin = st.sidebar.slider("Bilirubin (mg/dl)", 0.0, 30.0, 1.5)
    cholesterol = st.sidebar.number_input("Cholesterol", 100, 2000, 300)
    albumin = st.sidebar.slider("Albumin (gm/dl)", 0.0, 5.0, 3.5)
    copper = st.sidebar.number_input("Copper (ug/day)", 0, 600, 80)
    alk_phos = st.sidebar.number_input("Alk Phos (U/liter)", 0, 15000, 1500)
    sgot = st.sidebar.number_input("SGOT (U/ml)", 0, 500, 100)
    tryglicerides = st.sidebar.number_input("Tryglicerides (mg/dl)", 0, 600, 110)
    platelets = st.sidebar.number_input("Platelets", 0, 600, 250)
    prothrombin = st.sidebar.slider("Prothrombin (sn)", 9.0, 20.0, 10.5)
    
    # Kategorik Seçimler
    drug = st.sidebar.selectbox("İlaç", ["D-penicillamine", "Placebo"])
    sex = st.sidebar.selectbox("Cinsiyet", ["F", "M"])
    ascites = st.sidebar.selectbox("Asit", ["N", "Y"])
    hepatomegaly = st.sidebar.selectbox("Karaciğer Büyümesi", ["N", "Y"])
    spiders = st.sidebar.selectbox("Örümcek Anjiyom", ["N", "Y"])
    edema = st.sidebar.selectbox("Ödem Durumu", ["N", "S", "Y"])
    stage = st.sidebar.selectbox("Evre (Stage)", [1.0, 2.0, 3.0, 4.0])

    data = {
        'N_Days': n_days, 'Age': age, 'Drug': drug, 'Sex': sex, 
        'Ascites': ascites, 'Hepatomegaly': hepatomegaly, 'Spiders': spiders, 
        'Edema': edema, 'Bilirubin': bilirubin, 'Cholesterol': cholesterol,
        'Albumin': albumin, 'Copper': copper, 'Alk_Phos': alk_phos, 'SGOT': sgot,
        'Tryglicerides': tryglicerides, 'Platelets': platelets, 'Prothrombin': prothrombin, 'Stage': stage
    }
    return pd.DataFrame([data])

input_df = get_user_inputs()

# 4. Veri Ön İşleme ve Özellik Mühendisliği (Hata Veren Kısım Burasıydı)
def preprocess_data(df, feature_order):
    df = df.copy()
    
    # Notebook'ta yaptığınız tüm yeni özellik türetme işlemlerini buraya ekliyoruz:
    df['Age_Years'] = df['Age'] / 365.25
    df['Bilirubin_Albumin_Ratio'] = df['Bilirubin'] / (df['Albumin'] + 0.001)
    df['Albumin_Bilirubin_Ratio'] = df['Albumin'] / (df['Bilirubin'] + 0.001)
    df['AST_ALT_Ratio'] = df['SGOT'] / (df['Alk_Phos'] + 1) # Not: ALT yoksa bu oranı Notebook'taki gibi düzenleyin
    df['Enzyme_Index'] = df['SGOT'] * df['Alk_Phos']
    df['Age_Bilirubin'] = df['Age_Years'] * df['Bilirubin']
    df['High_Bilirubin'] = (df['Bilirubin'] > 1.2).astype(int)

    # Logaritmik Dönüşümler
    log_cols = ['Bilirubin', 'Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides']
    for col in log_cols:
        df[f'log_{col}'] = np.log1p(df[col])

    # One-Hot Encoding Taklidi (Modelin beklediği Y/N sütunları)
    df['Drug_Placebo'] = (df['Drug'] == 'Placebo').astype(int)
    df['Sex_M'] = (df['Sex'] == 'M').astype(int)
    df['Ascites_Y'] = (df['Ascites'] == 'Y').astype(int)
    df['Hepatomegaly_Y'] = (df['Hepatomegaly'] == 'Y').astype(int)
    df['Spiders_Y'] = (df['Spiders'] == 'Y').astype(int)
    df['Edema_S'] = (df['Edema'] == 'S').astype(int)
    df['Edema_Y'] = (df['Edema'] == 'Y').astype(int)

    # Kategorik Mapping (Gerekli olanlar için)
    binary_map = {'N': 0, 'Y': 1, 'F': 0, 'M': 1, 'Placebo': 0, 'D-penicillamine': 1}
    for col in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders']:
        df[col] = df[col].map(binary_map)
    
    df['Edema'] = df['Edema'].map({'N': 0, 'S': 1, 'Y': 2})

    # KRİTİK: Sadece modelin eğitildiği sütunları ve O SIRALAMA ile döndür
    return df[feature_order]

# 5. Arayüz ve Tahmin
st.subheader("📊 Girilen Veriler")
st.dataframe(input_df)

if st.button("🚀 Tahmin Et"):
    # Veriyi modelin beklediği formata getir
    try:
        final_df = preprocess_data(input_df, assets['feature_order'])
        
        # Olasılık Tahminleri (Ağırlıklı Harmanlama)
        p_cat = assets['cat_model'].predict_proba(final_df)
        p_xgb = assets['xgb_model'].predict_proba(final_df)
        p_lgbm = assets['lgbm_model'].predict_proba(final_df)
        
        # Notebook'taki oranlar: %50 Cat, %25 XGB, %25 LGBM
        blended_probs = (p_cat * 0.50) + (p_xgb * 0.25) + (p_lgbm * 0.25)
        
        # Sonuçları Göster
        st.markdown("---")
        cols = st.columns(3)
        target_classes = ["Yaşıyor (C)", "Nakil (CL)", "Vefat (D)"]
        
        for i in range(3):
            cols[i].metric(target_classes[i], f"% {blended_probs[0][i]*100:.2f}")

        # En yüksek olasılığı vurgula
        res_idx = np.argmax(blended_probs)
        st.success(f"En Yüksek Olasılık: **{target_classes[res_idx]}**")
        
    except KeyError as e:
        st.error(f"Hata: Eksik sütun tespit edildi. Lütfen Notebook'taki özellik türetme adımlarını kontrol edin. {e}")