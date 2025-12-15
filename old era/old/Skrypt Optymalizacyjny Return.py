def objective_max_return(trial):
    # Parametry (te same)
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
    wf_ret = []
    
    splits = walk_forward_split(full_df, n_splits=N_SPLITS)
    
    for train_subset, test_subset in splits:
        # ... (Skalowanie i Model jak wyżej) ...
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_subset[features])
        train_sc = train_subset.copy(); test_sc = test_subset.copy()
        train_sc[features] = scaler.transform(train_subset[features])
        test_sc[features]  = scaler.transform(test_subset[features])
        
        X_train, y_train = create_sequences(train_sc, features, 'target', params['lookback'])
        X_test, _        = create_sequences(test_sc, features, 'target', params['lookback'])
        
        if len(X_train) < 100 or len(X_test) < 20: return -100.0
            
        model = build_model((params['lookback'], len(features)), params)
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                  verbose=0, callbacks=[es], shuffle=False)
        
        # ... (Predykcja i Logika jak wyżej) ...
        preds = model.predict(X_test, verbose=0).flatten()
        test_data = test_subset.iloc[params['lookback']:].copy()
        market_ret = test_data['log_return'].shift(-1).fillna(0).values
        trend = test_data['Trend_Up'].values
        
        final_pos = []
        for i in range(len(preds)):
            sig = 0
            if preds[i] > params['t_long']: sig = 1
            elif preds[i] < params['t_short']: sig = -1
            if sig == 1 and trend[i] == 0: sig = 0
            if sig == -1 and trend[i] == 1: sig = 0
            final_pos.append(sig)
            
        final_pos = np.array(final_pos)
        strat_ret = final_pos * market_ret
        
        # --- Obliczanie RETURN ---
        n_trades = np.sum(final_pos != 0)
        
        # Kara za brak handlu (nawet przy returnie, 0 trade'ów to nie strategia)
        if n_trades < 10: 
            wf_ret.append(-20.0) 
            continue
            
        total_return = (np.exp(np.sum(strat_ret)) - 1) * 100
        wf_ret.append(total_return)
        
        del model
        tf.keras.backend.clear_session()

    # Wynik to średni zwrot roczny (Average Annual Return)
    score = np.mean(wf_ret)
    print(f"Trial ROI: {score:.2f}%")
    return score