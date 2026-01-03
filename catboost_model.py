# catboost_model.py
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_and_select_data_interactive, choose_lease_term

def train_catboost_model_for_residues(csv_file):
    """
    Trainiert ein CatBoost-Modell für Residuen mit 2025 EV-Merkmalen.
    Nutzt die neuen technischen Spezifikationen und Marktfaktoren.
    """
    df_full = pd.read_csv(csv_file)

    # Simulation der SSM-Basisprognose für Trainingsdaten
    # Realistische EV-Abwertung: 25% Jahr 1, 15% Jahr 2, 10% Jahr 3
    def calculate_baseline_rv(neupreis, alter_monate):
        if alter_monate <= 12:
            return neupreis * (1 - 0.25 * (alter_monate/12))
        elif alter_monate <= 24:
            year1_val = neupreis * 0.75
            return year1_val * (1 - 0.15 * ((alter_monate-12)/12))
        else:
            year1_val = neupreis * 0.75
            year2_val = year1_val * 0.85
            return year2_val * (1 - 0.10 * ((alter_monate-24)/12))

    df_full['ssm_baseline_rv_prediction_train'] = df_full.apply(
        lambda row: max(0, calculate_baseline_rv(row['Neupreis_EUR'], row['Alter_Monate'])), axis=1
    )

    df_full['residue_train'] = df_full['Restwert_EUR'] - df_full['ssm_baseline_rv_prediction_train']

    # Neue Features für 2025
    features = [
        'Marke', 'Modell', 'Variante', 'Laufleistung_km', 'Batteriegroesse_kWh',
        'Karosserieform', 'Spitzenleistung_kW', 'WLTP_Energieverbrauch_kWh_100km',
        'WLTP_Elektrische_Reichweite_km', 'Farbe', 'Region',
        'Zustand_Skala_1_5', 'Marktfaktor_Elektropreis', 'Marktfaktor_EVNachfrage'
    ]

    categorical_features_indices = [
        features.index(col) for col in [
            'Marke', 'Modell', 'Variante', 'Karosserieform', 'Farbe', 'Region',
            'Marktfaktor_Elektropreis', 'Marktfaktor_EVNachfrage'
        ]
    ]

    X_train = df_full[features]
    y_train = df_full['residue_train']

    # CatBoost-Modell mit optimierten Parametern für EV-Daten
    model = CatBoostRegressor(
        iterations=1200,  # Mehr Iterationen für komplexe EV-Daten
        learning_rate=0.04,  # Niedrigere Lernrate für Stabilität
        depth=9,  # Tiefere Bäume für technische Merkmale
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        cat_features=categorical_features_indices,
    )

    train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)

    print("\n--- Training des CatBoost Modells für 2025 EV-Daten ---")
    print("Features: Karosserieform, Spitzenleistung, WLTP-Verbrauch, WLTP-Reichweite, Elektropreis-Marktfaktor")
    model.fit(train_pool, verbose=False)
    print("CatBoost Modelltraining für 2025 EV-Spezifikationen abgeschlossen.")

    global_feature_importances = model.get_feature_importance()

    return model, features, categorical_features_indices, global_feature_importances

def run_catboost_prediction(catboost_model, kalman_output, selected_data, catboost_features, catboost_cat_indices, global_feature_importances):
    """
    Führt CatBoost-Residuenkorrektur mit 2025 EV-Merkmalen durch.
    """
    selected_model_name = selected_data["selected_model_name"]
    chosen_lease_term = kalman_output["forecast_months"]
    ssm_baseline_rv_prediction = kalman_output["ssm_baseline_rv_prediction"]

    # Vorbereitung der 2025 EV-Features für CatBoost
    X_predict = pd.DataFrame([selected_data["full_row_data"][catboost_features].values], columns=catboost_features)

    # Residuum-Vorhersage
    residue_prediction = catboost_model.predict(X_predict)[0]

    # Finale Restwertberechnung
    final_predicted_rv = ssm_baseline_rv_prediction + residue_prediction

    # Ergebnisse für HCBE
    print("\n--- 3. Schritt: CatBoost Residuen-Korrektur mit 2025 EV-Merkmalen ---")
    print(f"Fahrzeug: {selected_model_name}")
    print(f"Laufzeit: {chosen_lease_term} Monate")
    print(f"Basis-Restwertprognose (Kalman Filter): {ssm_baseline_rv_prediction:.2f} €")
    print(f"Korrektur durch 2025 EV-Faktoren (CatBoost): {residue_prediction:.2f} €")
    print(f"--> Finaler prognostizierter Restwert: {final_predicted_rv:.2f} €")

    # Feature Importances für 2025 EV-Merkmale
    feature_names = catboost_features
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': global_feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Mapping der Feature-Namen für bessere Darstellung
    feature_display_names = {
        'Spitzenleistung_kW': 'Spitzenleistung (kW)',
        'WLTP_Energieverbrauch_kWh_100km': 'WLTP Verbrauch (kWh/100km)',
        'WLTP_Elektrische_Reichweite_km': 'WLTP Reichweite (km)',
        'Karosserieform': 'Karosserieform',
        'Marktfaktor_Elektropreis': 'Marktfaktor Elektropreis',
        'Batteriegroesse_kWh': 'Batteriegröße (kWh)',
        'Laufleistung_km': 'Laufleistung (km)',
        'Zustand_Skala_1_5': 'Fahrzeugzustand',
        'Marktfaktor_EVNachfrage': 'EV-Nachfrage',
        'Region': 'Region',
        'Modell': 'Modell',
        'Variante': 'Variante',
        'Marke': 'Marke',
        'Farbe': 'Farbe'
    }

    importance_df['Feature_Display'] = importance_df['Feature'].map(feature_display_names).fillna(importance_df['Feature'])

    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(10),
               x='Importance',
               y='Feature_Display',
               hue='Feature_Display',  # Explizite hue-Zuweisung
               palette='viridis',
               legend=False)           # Legende unterdrücken
    plt.title(f"Top 10 Einflussfaktoren der 2025 EV-Residuen-Korrektur\n{selected_model_name} (Laufzeit {chosen_lease_term} Monate)")
    plt.xlabel('Wichtigkeit (CatBoost Feature Importance)')
    plt.ylabel('2025 EV-Merkmal')
    plt.tight_layout()
    plt.show()

    # Zusätzliche Analyse der technischen Merkmale
    print("\n--- Analyse der technischen 2025 EV-Merkmale ---")
    row_data = selected_data["full_row_data"]
    print(f"Spitzenleistung: {row_data['Spitzenleistung_kW']:.0f} kW")
    print(f"WLTP Energieverbrauch: {row_data['WLTP_Energieverbrauch_kWh_100km']:.1f} kWh/100km")
    print(f"WLTP Elektrische Reichweite: {row_data['WLTP_Elektrische_Reichweite_km']:.0f} km")
    print(f"Karosserieform: {row_data['Karosserieform']}")
    print(f"Marktfaktor Elektropreis: {row_data['Marktfaktor_Elektropreis']}")

    return final_predicted_rv

if __name__ == "__main__":
    csv_path = "hyundai_ev_restwerte_2025_aktualisiert.csv"

    # Offline-Phase: CatBoost-Modell für 2025 EV-Daten trainieren
    trained_catboost_model, catboost_features, catboost_cat_indices, global_feature_importances = train_catboost_model_for_residues(csv_path)

    # Simulation der Online-Anfrage
    print("\n--- Simulation der Online-Anfrage mit 2025 EV-Daten ---")
    selected_data_for_test = load_and_select_data_interactive(csv_path)
    if selected_data_for_test:
        chosen_term_for_test = choose_lease_term()

        # Simulierter Kalman-Output für Test
        simulated_kalman_output = {
            "ssm_baseline_rv_prediction": selected_data_for_test["initial_rv"] * (1 - 0.021 * chosen_term_for_test),
            "forecast_months": chosen_term_for_test,
            "selected_model_name": selected_data_for_test["selected_model_name"]
        }
        simulated_kalman_output["ssm_baseline_rv_prediction"] = max(0, simulated_kalman_output["ssm_baseline_rv_prediction"])

        final_rv_catboost_test = run_catboost_prediction(
            trained_catboost_model,
            simulated_kalman_output,
            selected_data_for_test,
            catboost_features,
            catboost_cat_indices,
            global_feature_importances
        )
        print(f"\nFinaler RV mit 2025 EV-Merkmalen: {final_rv_catboost_test:.2f} €")
