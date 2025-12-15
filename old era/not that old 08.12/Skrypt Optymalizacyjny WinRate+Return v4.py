import numpy as np
import pandas as pd
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1. ŁADOWANIE DANYCH (GLOBALNE) - Tutaj naprawiamy Twój błąd!
# =============================================================================

# Ustawienia
LOOKBACK = 21   # Twój wymagany lookback
TARGET_COL = 'target' # Nazwa kolumny, którą przewidujemy (0 lub 1)

# --- OPCJA A: Jeśli masz plik CSV (ODKOMENTUJ I UŻYJ TEGO) ---
"""
# Wczytaj dane
df = pd.read_csv('twoja_sciezka_do_pliku.csv')

# Tutaj zrób swój preprocessing (np. usunięcie NaN, wskaźniki)
# df = df.dropna()

# Skalowanie (ważne dla LSTM!)
scaler = MinMaxScaler()
# Zakładamy, że skalujemy wszystko oprócz targetu, lub wszystko łącznie
feature_cols = [c for c in df.columns if c != TARGET_COL]
scaled_features = scaler.fit_transform(df[feature_cols])
targets = df[TARGET_COL].values

# Funkcja tworząca sekwencje (okna czasowe)
def create_sequences(features, targets, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i+lookback])
        y.append(targets[i+lookback])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, targets, LOOKBACK)

# Podział chronologiczny (nie tasujemy, żeby nie zaglądać w przyszłość)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
"""

# --- OPCJA B: DANE TESTOWE (Żeby kod działał od razu po wklejeniu) ---
# (Jeśli masz już swoje dane wyżej, zakomentuj tę sekcję B)
print("Generuję dane testowe (zastąp je swoim plikiem CSV)...")
dummy_samples = 2000
dummy_features = 5
# Generujemy losowe przebiegi czasowe
X_dummy = np.random.rand(dummy_samples, LOOKBACK, dummy_features)
# Generujemy losowy target (0 lub 1)
y_dummy = np.random.randint(0, 2, size=dummy_samples)

# Podział na trening i walidację
X_train, X_val, y_train, y_val = train_test_split(
    X_dummy, y_dummy, test_size=0.2, shuffle=False
)
print(f"Dane załadowane! Kształt X_train: {X_train.shape}")
# -----------------------------------------------------------------------------

# Sprawdzenie bezpieczeństwa (czy na pewno mamy dobre wymiary)
assert X_train.shape[1] == 21, "BŁĄD: Lookback musi wynosić 21!"

# =============================================================================
# 2. FUNKCJA CELU (STRATEGIA SNAJPERA)
# =============================================================================
def objective(trial):
    # Teraz funkcja WIDZI zmienne X_train, y_train bo są zdefiniowane wyżej w skrypcie
    
    # 1. Hiperparametry
    neurons = trial.suggest_int('neurons', 60, 100, step=5)
    dropout = trial.suggest_float('dropout', 0.10, 0.25)
    lr = trial.suggest_float('lr', 0.0005, 0.0015, log=True)
    
    # Threshold Dist: Jak bardzo pewny musi być model (0.5 +/- dist)
    threshold_dist = trial.suggest_float('threshold_dist', 0.02, 0.08)
    
    batch_size = 16 # Ustalone na sztywno jako dobre

    # 2. Budowa modelu
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 3. Trening z zabezpieczeniem (Early Stopping)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, # Dajemy dużo, ES utnie wcześniej
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    # 4. Ewaluacja "Snajpera"
    # Pobieramy surowe prawdopodobieństwa dla zbioru walidacyjnego
    raw_preds = model.predict(X_val, verbose=0).flatten()
    
    # Definiujemy progi decyzyjne
    upper_threshold = 0.5 + threshold_dist
    lower_threshold = 0.5 - threshold_dist
    
    # Decyzja: Gramy tylko gdy pewność jest wysoka
    # Maska (True/False) wskazująca gdzie podjęliśmy decyzję
    action_mask = (raw_preds > upper_threshold) | (raw_preds < lower_threshold)
    
    # Statystyki
    total_samples = len(y_val)
    actions_count = np.sum(action_mask)
    action_rate = actions_count / total_samples
    
    # Zapisujemy atrybuty do analizy później
    trial.set_user_attr("action_rate", action_rate)
    
    # --- WARUNEK KRYTYCZNY (FILTR) ---
    # Jeśli model gra rzadziej niż w 10% sytuacji -> ODRZUCAMY GO (wynik 0)
    if action_rate < 0.10:
        return 0.0

    # Obliczamy Win Rate tylko dla podjętych akcji
    if actions_count == 0:
        real_winrate = 0.0
    else:
        # Predykcja klasy (0 lub 1) dla aktywnych
        # Dla long/short: >0.5 to 1, <0.5 to 0
        active_preds_class = (raw_preds[action_mask] > 0.5).astype(int)
        active_actuals = y_val[action_mask]
        
        real_winrate = accuracy_score(active_actuals, active_preds_class) * 100

    return real_winrate

# =============================================================================
# 3. URUCHOMIENIE OPTUNY
# =============================================================================
if __name__ == "__main__":
    print("Rozpoczynam optymalizację...")
    
    # TPESampler - standardowy, wydajny sampler
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Uruchamiamy 50 prób
    study.optimize(objective, n_trials=50)

    print("\n--- WYNIKI ---")
    best = study.best_trial
    print(f"Najlepszy Win Rate: {best.value:.2f}%")
    print(f"Przy Action Rate:   {best.user_attrs['action_rate']*100:.2f}%")
    print("Najlepsze parametry:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    
    # Wyświetlenie topowych modeli spełniających kryteria
    df = study.trials_dataframe()
    # Filtrujemy te z wynikiem > 57.5% i action rate > 10% (co już jest w value > 0)
    good_models = df[df['value'] >= 57.5].sort_values(by='value', ascending=False)
    
    if not good_models.empty:
        print("\nLista modeli spełniających kryterium > 57.5% WinRate:")
        cols = ['number', 'value', 'user_attrs_action_rate', 'params_threshold_dist', 'params_neurons']
        print(good_models[cols].head())
    else:
        print("\nŻaden model nie osiągnął progu 57.5% przy wymaganej aktywności.")