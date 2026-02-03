import requests

BASE_URL = "http://localhost:8000"

# Örnek 1: Yapılandırılmış veri ile tahmin
print("=" * 50)
print("Örnek 1: Yapılandırılmış veri ile tahmin")
print("=" * 50)

response = requests.post(f"{BASE_URL}/predict", 
    json={
        "yas": 72,
        "anemi": 0,
        "kreatinin_fosfokinaz": 1205,
        "diyabet": 1,
        "ejeksiyon_fraksiyonu": 23,
        "yuksek_kan_basinci": 0,
        "trombositler": 315052,
        "serum_kreatinin": 2.5,
        "serum_sodyum": 105,
        "cinsiyet": 1,
        "sigara": 0,
        "zaman": 5
    }
)

print(f"Durum Kodu: {response.status_code}")
print(f"Sonuç: {response.json()}")

# Örnek 2: Ham liste ile tahmin
print("\n" + "=" * 50)
print("Örnek 2: Ham liste ile tahmin")
print("=" * 50)

# [yaş, anemi, kreatinin_fosfokinaz, diyabet, ejeksiyon_fraksiyonu, 
#  yüksek_kan_basıncı, trombositler, serum_kreatinin, serum_sodyum, 
#  cinsiyet, sigara, zaman]
response_raw = requests.post(f"{BASE_URL}/predict_raw", 
    json=[72, 0, 1205, 1, 23, 0, 315052, 2.5, 105, 1, 0, 5]
)

print(f"Durum Kodu: {response_raw.status_code}")
print(f"Sonuç: {response_raw.json()}")

# Örnek 3: Farklı bir hasta verisi
print("\n" + "=" * 50)
print("Örnek 3: Farklı hasta verisi")
print("=" * 50)

response2 = requests.post(f"{BASE_URL}/predict", 
    json={
        "yas": 55,
        "anemi": 0,
        "kreatinin_fosfokinaz": 582,
        "diyabet": 0,
        "ejeksiyon_fraksiyonu": 38,
        "yuksek_kan_basinci": 0,
        "trombositler": 263358,
        "serum_kreatinin": 1.1,
        "serum_sodyum": 140,
        "cinsiyet": 1,
        "sigara": 0,
        "zaman": 100
    }
)

print(f"Durum Kodu: {response2.status_code}")
print(f"Sonuç: {response2.json()}")