def objective_max_winrate(trial):
    # Parametry (takie same jak wcześniej)
    params = {
        'lookback': trial.suggest_int('lookback', *SEARCH_SPACE['lookback']),
        'neurons':  trial.suggest_int('neurons', *SEARCH_SPACE['neurons']),
        'dropout':  trial.suggest_float('dropout', *SEARCH_SPACE['dropout']),
        'lr':       trial.suggest_float('lr', *SEARCH_SPACE['lr'], log=True),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
        'epochs':   trial.suggest_int('epochs', *SEARCH_SPACE['epochs']),
        't_long':   trial.suggest_float('t_long', *SEARCH_SPACE['thresh_long']),
        't_short':  trial.suggest_float('t_short', *SEARCH_SPACE['thresh_short'])
    }
    
    features = ['log_return', 'volatility_30', 'dist_high', 'dist_low', 'momentum_3d']
    wf_wr = [] # Zbieramy tylko Win Rate
    
    splits = walk_forward_split(full_df, n_splits=N_SPLITS)
    
    for train_subset, test_subset in splits:
        # --- Standardowa procedura (Skalowanie -> Model -> Trening) ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_subset[features])
        train_sc = train_subset.copy(); test_sc = test_subset.copy()
        train_sc[features] = scaler.transform(train_subset[features])
        test_sc[features]  = scaler.transform(test_subset[features])
        
        X_train, y_train = create_sequences(train_sc, features, 'target', params['lookback'])
        X_test, _        = create_sequences(test_sc, features, 'target', params['lookback'])
        
        if len(X_train) < 100 or len(X_test) < 20: return 0.0
            
        model = build_model((params['lookback'], len(features)), params) # Używamy funkcji build_model z poprzedniego kodu
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                  verbose=0, callbacks=[es], shuffle=False)
        
        # --- Logika Decyzyjna ---
        preds = model.predict(X_test, verbose=0).flatten()
        test_data = test_subset.iloc[params['lookback']:].copy()
        market_ret = test_data['log_return'].shift(-1).fillna(0).values
        trend = test_data['Trend_Up'].values
        
        final_pos = []
        for i in range(len(preds)):
            sig = 0
            if preds[i] > params['t_long']: sig = 1
            elif preds[i] < params['t_short']: sig = -1
            
            # Filtr trendu
            if sig == 1 and trend[i] == 0: sig = 0
            if sig == -1 and trend[i] == 1: sig = 0
            final_pos.append(sig)
            
        final_pos = np.array(final_pos)
        
        # --- Obliczanie Win Rate ---
        n_trades = np.sum(final_pos != 0)
        
        # KARA ZA BRAK AKTYWNOŚCI (Kluczowe dla Win Rate)
        if n_trades < 15: 
            wf_wr.append(0.0) # Traktujemy jako porażkę
            continue
            
        wins = np.sum(np.sign(final_pos[final_pos!=0]) == np.sign(market_ret[final_pos!=0]))
        win_rate = (wins / n_trades) * 100
        wf_wr.append(win_rate)
        
        del model
        tf.keras.backend.clear_session()

    # Wynik to średni Win Rate ze wszystkich lat
    score = np.mean(wf_wr)
    print(f"Trial WR: {score:.2f}%")
    return score