# Property Price Prediction System 🏠

System AI do przewidywania cen mieszkań w Polsce przy użyciu machine learning.

## 📋 Opis projektu

Systema jest zbudowany w celu:
- **Analizy** danych dotyczących cen nieruchomości w Polsce
- **Trenowania** modelu machine learning (Random Forest) na historycznych danych
- **Udostępnienia** modelu jako usługi webowej (API) dla predykcji cen
- **Filtrowania** właściwości po województwie, mieście i powiecie

## 🎯 Cechy systemu

✅ **Model Machine Learning** - Random Forest Regressor  
✅ **API REST** - FastAPI z automatyczną dokumentacją  
✅ **Predykcja cen** - na podstawie charakterystyk nieruchomości  
✅ **Filtrowanie danych** - po województwie, mieście, powiecie  
✅ **Walidacja danych** - Pydantic models  
✅ **Dokumentacja** - Swagger/OpenAPI  

## 📊 Dane treningowe

- **Liczba próbek**: 24,181 nieruchomości
- **Liczba cech**: 8 (powierzchnia, liczba pokoi, rok budowy, typ ogrzewania, materiał budynku, typ budynku, rynek, województwo)
- **Województwa**: 16 polskich województw
- **Zakresy cen**: 56,396 PLN - 1,377,242 PLN

## 🤖 Model

**Typ**: Regresja (przewidywanie wartości numerycznej)  
**Algorytm**: Random Forest Regressor (100 drzew decyzyjnych)  

### Wydajność modelu:
- **R² Score**: 0.6364 (63.64% wariancji wyjaśnionej)
- **RMSE**: 148,161 PLN (średni błąd przewidywania)
- **MAE**: 112,492 PLN (średnia błąd absolutny)
- **Train set**: 19,344 próbek
- **Test set**: 4,837 próbek

## 🛠️ Wymagania

- Python 3.10+
- Narzędzie `uv` do zarządzania zależnościami
- Git do kontroli wersji

## 📦 Instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/Arekfl/ceny_nieruchomosci_AI.git
cd ceny_nieruchomosci_AI
```

### 2. Przygotowanie środowiska wirtualnego z `uv`

```bash
# Instalacja uv (jeśli jeszcze nie zainstalowany)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Dodanie uv do PATH
export PATH="$HOME/.local/bin:$PATH"

# Tworzenie i aktywacja wirtualnego środowiska
uv venv
source .venv/bin/activate  # Linux/Mac
```

### 3. Instalacja zależności

```bash
pip install pandas numpy scikit-learn fastapi uvicorn pydantic joblib python-dotenv
```

## 🚀 Uruchomienie serwera

### Opcja 1: Bezpośrednio z Pythona

```bash
python run_server.py
```

### Opcja 2: Za pomocą uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Serwer będzie dostępny pod adresem:
- **API**: http://localhost:8000
- **Dokumentacja**: http://localhost:8000/docs

## 📡 API Endpoints

### 1. Predykcja ceny
```http
POST /predict
```

**Request**:
```json
{
  "area": 120.5,
  "rooms": 4,
  "year_constructed": 2020,
  "heating": "gazowe",
  "building_material": "cegła",
  "building_type": "bliźniak",
  "market": "pierwotny",
  "voivodeship": "mazowieckie"
}
```

**Response**:
```json
{
  "predicted_price": 485250.50,
  "currency": "PLN",
  "confidence": "High",
  "input_features": { ... }
}
```

### 2. Informacje o modelu
```http
GET /info
```

### 3. Health check
```http
GET /health
```

### 4. Filtrowanie właściwości
```http
GET /filter?voivodeship=mazowieckie
```

## 🧪 Testowanie API

```bash
# Upewnij się że serwer jest uruchomiony
python run_server.py

# W innym terminalu uruchom testy
python test_api.py
```

## 📁 Struktura projektu

```
ceny_nieruchomosci_AI/
├── app/                    # Aplikacja FastAPI
│   ├── main.py            # Główne endpointy
│   ├── models.py          # Pydantic models
│   └── config.py          # Konfiguracja
├── data/
│   ├── raw/               # Surowe dane
│   └── processed/         # Czyste dane
├── models/                # Wytrenowane modele
│   ├── price_model.joblib
│   ├── label_encoders.joblib
│   └── features.joblib
├── notebooks/             # Jupyter notebooks
├── pyproject.toml         # Zależności
├── run_server.py          # Uruchomienie serwera
├── test_api.py            # Testy
└── README.md              # Dokumentacja
```

## 🔧 Technologie

- **Python 3.10** - Język programowania
- **pandas, numpy** - Przetwarzanie danych
- **scikit-learn** - Machine Learning (Random Forest)
- **FastAPI** - Framework API
- **Uvicorn** - ASGI server
- **Pydantic** - Walidacja danych
- **joblib** - Serializacja modelu

## 📝 Licencja

MIT License

---

**Wersja**: 1.0.0  
**Data**: Luty 2025
