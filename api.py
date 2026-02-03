from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# FastAPI uygulaması oluştur
app = FastAPI(
    title="Heart Failure Prediction API",
    description="KNN modeli ile kalp yetmezliği tahmini yapan API",
    version="1.0.0"
)

# Modeli yükle
model = joblib.load("knn_model.joblib")

# İstek için veri modeli
class HeartFailureInput(BaseModel):
    yas: float                      # age
    anemi: int                      # anaemia (0 veya 1)
    kreatinin_fosfokinaz: float     # creatinine_phosphokinase
    diyabet: int                    # diabetes (0 veya 1)
    ejeksiyon_fraksiyonu: float     # ejection_fraction
    yuksek_kan_basinci: int         # high_blood_pressure (0 veya 1)
    trombositler: float             # platelets
    serum_kreatinin: float          # serum_creatinine
    serum_sodyum: float             # serum_sodium
    cinsiyet: int                   # sex (0 veya 1)
    sigara: int                     # smoking (0 veya 1)
    zaman: int                      # time (gün)

# Yanıt modeli
class PredictionResponse(BaseModel):
    tahmin: int
    durum: str
    mesaj: str

@app.get("/")
def root():
    return {"mesaj": "Heart Failure Prediction API'ye hoş geldiniz!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HeartFailureInput):
    """
    Kalp yetmezliği tahmini yapar.
    
    - **tahmin**: 0 = Ölüm, 1 = Yaşam
    - **durum**: Tahmin sonucunun açıklaması
    """
    # Girdi verilerini numpy array'e dönüştür
    features = np.array([[
        input_data.yas,
        input_data.anemi,
        input_data.kreatinin_fosfokinaz,
        input_data.diyabet,
        input_data.ejeksiyon_fraksiyonu,
        input_data.yuksek_kan_basinci,
        input_data.trombositler,
        input_data.serum_kreatinin,
        input_data.serum_sodyum,
        input_data.cinsiyet,
        input_data.sigara,
        input_data.zaman
    ]])
    
    # Tahmin yap
    prediction = model.predict(features)[0]
    
    # Sonucu döndür
    durum = "Yaşam" if prediction == 0 else "Ölüm"
    
    return PredictionResponse(
        tahmin=int(prediction),
        durum=durum,
        mesaj=f"Model tahmini: {durum}"
    )

@app.post("/predict_raw")
def predict_raw(data: list[float]):
    """
    Ham liste formatında veri ile tahmin yapar.
    12 özellik sırasıyla: [yaş, anemi, kreatinin_fosfokinaz, diyabet, 
    ejeksiyon_fraksiyonu, yüksek_kan_basıncı, trombositler, serum_kreatinin, 
    serum_sodyum, cinsiyet, sigara, zaman]
    """
    if len(data) != 12:
        return {"hata": "12 özellik gerekli"}
    
    features = np.array([data])
    prediction = model.predict(features)[0]
    durum = "Yaşam" if prediction == 0 else "Ölüm"
    
    return {
        "tahmin": int(prediction),
        "durum": durum
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
