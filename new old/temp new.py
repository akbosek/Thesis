import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# Importy Keras / TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adamax, RMSprop, SGD, Nadam, Adagrad, Adadelta, Ftrl

# ==============================================================================
# 1. KONFIGURACJA HIPERPARAMETRÓW (TWOJE CENTRUM STEROWANIA)
# ==============================================================================
# Tutaj zmieniasz ustawienia, żeby dostroić model. Czytaj komentarze!

# --- PLIKI I KATALOGI ---
DATA_FILE      = 'bitcoin_2018_feb_data.csv'  # Nazwa pliku z danymi wejściowymi
VAL_START_DATE = '2022-01-01'                 # Data odcięcia: Wszystko PO tej dacie to test (Walidacja)
OUTPUT_DIR     = 'RAPORT_FINALNY_KOMENTARZE'  # Folder, gdzie zapiszą się wykresy i model

# --- TRENING (Ogólne) ---
HP_SEED        = 42      # Ziarno losowości. 42 = powtarzalne wyniki. Zmień na inną liczbę, by sprawdzić czy to nie przypadek.
HP_EPOCHS      = 100     # Maksymalna liczba pętli nauki. (EarlyStopping i tak przerwie wcześniej, jeśli nie będzie postępów).
HP_BATCH_SIZE  = 16      # Wielkość paczki danych. 
                         # - Mniej (16-32): Dokładniejsza nauka, ale wolniejsza. Model częściej aktualizuje wagi.
                         # - Więcej (64-128): Szybsza nauka, stabilniejsza, ale może pominąć detale.
HP_LR          = 0.001   # Learning Rate (Szybkość Uczenia).
                         # - 0.001: Standardowy start.
                         # - 0.0001: Dla precyzyjnego dostrajania (gdy model skacze po wykresie).

# --- ARCHITEKTURA SIECI (Mózg Modelu) ---
HP_LOOKBACK    = 30      # Pamięć wsteczna (ile dni widzi model).
                         # - Krótko (7-14): Szybka reakcja, ale dużo szumu.
                         # - Długo (30-60): Widzi trendy, ale może reagować z opóźnieniem (lag).

HP_LSTM_1      = 128      # Liczba neuronów w 1. warstwie LSTM.
                         # - Zwiększ (np. 64, 128): Jeśli model jest "za głupi" i nie widzi zależności (Accuracy ~50%).
                         # - Zmniejsz (np. 16): Jeśli model "wkuwa na pamięć" (Overfitting) i traci na walidacji.

HP_LSTM_2      = 64      # Liczba neuronów w 2. warstwie LSTM (zazwyczaj połowa pierwszej).

HP_DENSE       = 16      # Neurony w warstwie "wnioskowania" (po LSTM, przed decyzją).
                         # - 8: Wymusza proste reguły (dobre na szum, mniejszy overfitting).
                         # - 16-32: Pozwala na bardziej złożoną logikę (lepsze łączenie faktów),
                         #          ale ryzykujesz, że model nauczy się przypadków na pamięć.

# --- REGULARYZACJA (Bezpieczniki przeciw Przeuczeniu) ---
HP_DROPOUT     = 0.2     # Ile % neuronów losowo wyłączyć w każdej rundzie.
                         # - 0.0: Brak wyłączania (Ryzyko Overfittingu).
                         # - 0.2 - 0.5: Standard. Zmusza model do szukania ogólnych reguł.
                         # - Zwiększ, jeśli na Treningu masz 1000% zysku, a na Walidacji stratę.

HP_L2          = 0.005   # "Podatek" od skomplikowania wag (L2 Regularization).
                         # - 0.001: Łagodny.
                         # - 0.01: Surowy. Mocno karze model za wkuwanie szumu. Zwiększ przy Overfittingu.

# --- WYBÓR OPTYMALIZATORA (Silnik) ---
# Odkomentuj (usuń #) przy tym, którego chcesz użyć. Tylko jeden może być aktywny!

# HP_OPTIMIZER = Adam(learning_rate=HP_LR)       # Standard. Dobry start, szybki.
HP_OPTIMIZER = Adamax(learning_rate=HP_LR)     # ZALECANY DO KRYPTO. Stabilny, ignoruje nagłe szpilki cenowe.
# HP_OPTIMIZER = RMSprop(learning_rate=HP_LR)    # Klasyk do sieci rekurencyjnych (LSTM).
# HP_OPTIMIZER = SGD(learning_rate=HP_LR, momentum=0.9) # Trudny w obsłudze, ale czasem daje najlepsze wyniki końcowe.
# HP_OPTIMIZER = Nadam(learning_rate=HP_LR)      # Adam z "dopalaczem" (Nesterov). Uczy się szybciej, ale w krypto łatwiej o Overfitting.
# HP_OPTIMIZER = Adagrad(learning_rate=HP_LR)    # Dobry do rzadkich danych (dużo zer), ale ma tendencję do zbyt szybkiego "gaśnięcia" nauki.
# HP_OPTIMIZER = Adadelta(learning_rate=HP_LR)   # Ulepszony Adagrad. Rozwiązuje problem "umierania" procesu nauki, ale zazwyczaj wolniejszy od Adama.
# HP_OPTIMIZER = Ftrl(learning_rate=HP_LR)       # Zaprojektowany do gigantycznych, rzadkich danych (np. kliknięcia w reklamy). W LSTM rzadko przydatny.

# ==============================================================================
# 2. INICJALIZACJA
# ==============================================================================
os.environ['PYTHONHASHSEED'] = str(HP_SEED)
random.seed(HP_SEED)
np.random.seed(HP_SEED)
tf.random.set_seed(HP_SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 3. PRZYGOTOWANIE DANYCH
# ==============================================================================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Brak pliku {DATA_FILE} w katalogu roboczym!")

# Wczytanie danych
df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# Definicja cech i celu
features = ['BTC_Close', 'BTC_Volume', 'Mayer_Ratio', 'RSI']
target = 'Target'

print(f" > Cechy wejściowe: {features}")

# Podział chronologiczny na trening i walidację
train_df = df[df.index < VAL_START_DATE].copy()
val_df   = df[df.index >= VAL_START_DATE].copy()

# Skalowanie danych (Fit tylko na treningu, Transform na obu)
scaler = MinMaxScaler()
X_train_raw = scaler.fit_transform(train_df[features])
X_val_raw   = scaler.transform(val_df[features])

y_train_raw = train_df[target].values
y_val_raw   = val_df[target].values

# Funkcja tworząca sekwencje czasowe (sliding window)
def create_dataset(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_dataset(X_train_raw, y_train_raw, HP_LOOKBACK)
X_val, y_val     = create_dataset(X_val_raw, y_val_raw, HP_LOOKBACK)

# Obliczenie wag klas (jeśli mamy nierówną liczbę wzrostów i spadków)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {0: class_weights[0], 1: class_weights[1]}

# ==============================================================================
# 4. BUDOWA MODELU LSTM
# ==============================================================================
model = Sequential([
    Input(shape=(HP_LOOKBACK, len(features))),
    
    # Pierwsza warstwa LSTM
    LSTM(HP_LSTM_1, return_sequences=True, kernel_regularizer=l2(HP_L2)),
    BatchNormalization(),
    Dropout(HP_DROPOUT),
    
    # Druga warstwa LSTM
    LSTM(HP_LSTM_2, return_sequences=False, kernel_regularizer=l2(HP_L2)),
    BatchNormalization(),
    Dropout(HP_DROPOUT),
    
    # Warstwa gęsta (interpretacja)
    Dense(HP_DENSE, activation='relu', kernel_regularizer=l2(HP_L2)),
    
    # Warstwa wyjściowa (prawdopodobieństwo 0-1)
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=HP_OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================================================================
# 5. TRENING
# ==============================================================================
print(f"\n--- [START TRENINGU] ---")

checkpoint = ModelCheckpoint(
    f"{OUTPUT_DIR}/best_model.keras", 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=0
)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

reduce_lr  = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=0.000001
)

history = model.fit(
    X_train, y_train,
    epochs=HP_EPOCHS,
    batch_size=HP_BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=cw_dict,
    verbose=1
)

# ==============================================================================
# 6. KALIBRACJA PROGU
# ==============================================================================
print("\n--- [KALIBRACJA] ---")
# Ustalamy próg decyzyjny na podstawie mediany pewności modelu na zbiorze treningowym
train_probs = model.predict(X_train, verbose=0).flatten()
optimal_threshold = np.median(train_probs)
print(f" > Wyliczony próg decyzyjny (Mediana): {optimal_threshold:.4f}")

# ==============================================================================
# 7. RAPORTOWANIE I GENEROWANIE WYKRESÓW
# ==============================================================================
def generate_report(X, y_true, prices, name, threshold):
    print(f"\n>>> RAPORT: {name} <<<")
    probs = model.predict(X, verbose=0).flatten()
    
    # KROK 1: Decyzja surowa (0 lub 1) wg progu
    raw_preds = (probs > threshold).astype(int)
    
    # KROK 2: Konwersja na Pozycję (1 = Long, -1 = Short)
    positions = np.where(raw_preds == 1, 1, -1)
    
    # Metryki
    acc = accuracy_score(y_true, raw_preds)
    fpr, tpr, _ = roc_curve(y_true, probs)
    gini = 2 * auc(fpr, tpr) - 1
    cm = confusion_matrix(y_true, raw_preds)
    
    longs = np.sum(positions == 1)
    shorts = np.sum(positions == -1)
    
    # Symulacja Equity (Kapitału)
    equity = [100.0]
    market = [100.0]
    
    # Przesuwamy realne zwroty o -1, bo decyzja dzisiaj wpływa na wynik jutra
    real_returns = prices.pct_change().shift(-1).dropna()
    
    # Wyrównanie długości tablic
    min_len = min(len(positions), len(real_returns))
    active_positions = positions[:min_len]
    active_rets = real_returns.values[:min_len]
    dates = real_returns.index[:min_len]
    active_prices = prices.values[:min_len]
    
    for i in range(min_len):
        ret = active_rets[i]
        pos = active_positions[i]
        
        # Logika strategii: Jeśli Short (-1) i cena spada (-), mamy zysk (+)
        strategy_ret = pos * ret
        
        equity.append(equity[-1] * (1 + strategy_ret))
        market.append(market[-1] * (1 + ret))
        
    final_eq = equity[-1] - 100.0
    final_mkt = market[-1] - 100.0
    
    print(f" > Accuracy: {acc:.2%}")
    print(f" > Gini:     {gini:.4f}")
    print(f" > Decyzje:  LONG={longs}, SHORT={shorts}")
    print(f" > Wynik:    Model={final_eq:.2f}% vs Rynek={final_mkt:.2f}%")
    
    # --- WYKRES 1: Equity Curve ---
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity[1:], label=f'Model L/S ({final_eq:.0f}%)', color='blue', linewidth=2)
    plt.plot(dates, market[1:], label=f'Rynek ({final_mkt:.0f}%)', color='gray', linestyle='--', alpha=0.7)
    plt.title(f'{name} - Krzywa Kapitału (Symulacja)')
    plt.ylabel('Kapitał (Start=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{name}_1_Equity.png")
    plt.close()
    
    # --- WYKRES 2: Sygnały Wejścia na Cenie ---
    plt.figure(figsize=(14, 7))
    plt.plot(dates, active_prices, label='Cena BTC', color='black', alpha=0.5)
    
    # Filtrowanie indeksów do markerów
    b_idx = [i for i, x in enumerate(active_positions) if x == 1]
    s_idx = [i for i, x in enumerate(active_positions) if x == -1]
    
    if len(b_idx) > 0:
        plt.scatter(dates[b_idx], active_prices[b_idx], marker='^', color='green', s=30, label='LONG', zorder=5)
    if len(s_idx) > 0:
        plt.scatter(dates[s_idx], active_prices[s_idx], marker='v', color='red', s=30, label='SHORT', zorder=5)
    
    plt.title(f'{name} - Sygnały Wejścia')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{name}_2_Signals.png")
    plt.close()
    
    # --- WYKRES 3: Macierz Pomyłek (Confusion Matrix) ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['SHORT (Pred)', 'LONG (Pred)'], 
                yticklabels=['SPADEK (Fakt)', 'WZROST (Fakt)'])
    plt.title(f'{name} - Macierz Decyzji')
    plt.ylabel('Prawda')
    plt.xlabel('Predykcja Modelu')
    plt.savefig(f"{OUTPUT_DIR}/{name}_3_Matrix.png")
    plt.close()

    # --- WYKRES 4: Krzywa ROC/Lorenz ---
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC (Gini={gini:.2f})', color='purple', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Losowo')
    plt.title(f'{name} - Krzywa ROC')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{name}_4_Lorenz.png")
    plt.close()

    # --- WYKRES 5: Histogram Pewności ---
    plt.figure(figsize=(10, 5))
    plt.hist(probs, bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Próg ({threshold:.2f})')
    plt.title(f'{name} - Rozkład Pewności Modelu')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{name}_5_Confidence.png")
    plt.close()
    
    # --- WYKRES 6: Mapa Pozycji w Czasie ---
    plt.figure(figsize=(12, 3))
    plt.step(dates, active_positions, where='post', color='purple', linewidth=1)
    plt.yticks([-1, 1], ['SHORT', 'LONG'])
    plt.title(f'{name} - Mapa Pozycji (Kiedy gramy co)')
    plt.savefig(f"{OUTPUT_DIR}/{name}_6_Positions.png")
    plt.close()

# Pobranie oryginalnych cen do symulacji (odpowiednio przyciętych o lookback)
train_prices = train_df['BTC_Close'].iloc[HP_LOOKBACK:]
val_prices   = val_df['BTC_Close'].iloc[HP_LOOKBACK:]

# Generowanie raportów
generate_report(X_train, y_train, train_prices, "TRENING", optimal_threshold)
generate_report(X_val, y_val, val_prices, "WALIDACJA", optimal_threshold)

print(f"\n✅ ZAKOŃCZONO. Wszystkie wyniki i wykresy znajdują się w folderze: {OUTPUT_DIR}")