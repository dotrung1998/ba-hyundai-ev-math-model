# data_utils.py

import pandas as pd
import numpy as np
import random # Neu hinzugefügt für die Simulation

def load_and_select_data_interactive(csv_file):
    """
    Lädt CSV, führt Benutzer durch die Auswahl eines EV-Modells und einer Variante
    und gibt die ausgewählten Daten zurück.
    Aktualisiert für 2025 Datenstruktur mit neuen Merkmalen.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{csv_file}' wurde nicht gefunden.")
        print("Bitte stellen Sie sicher, dass die 'hyundai_ev_restwerte_2025_aktualisiert.csv' im selben Verzeichnis liegt.")
        return None

    # Marken auswählen
    unique_brands = df["Marke"].unique().tolist()
    print("Verfügbare Marken:")
    for i, brand in enumerate(unique_brands):
        print(f"{i+1}. {brand}")
    selected_brand = None
    while selected_brand is None:
        try:
            brand_choice_input = input("Marke per Nummer auswählen: ")
            brand_choice = int(brand_choice_input) - 1
            if 0 <= brand_choice < len(unique_brands):
                selected_brand = unique_brands[brand_choice]
                print(f"Sie haben ausgewählt: {selected_brand}")
            else:
                print("Ungültige Auswahl. Bitte geben Sie eine gültige Nummer ein.")
        except ValueError:
            print("Ungültige Eingabe. Bitte geben Sie eine Nummer ein.")

    # Modelle innerhalb der gewählten Marke auswählen
    models_in_brand = df[df["Marke"] == selected_brand]["Modell"].unique().tolist()
    print(f"\nVerfügbare Modelle für {selected_brand}:")
    for i, model in enumerate(models_in_brand):
        print(f"{i+1}. {model}")
    selected_model = None
    while selected_model is None:
        try:
            model_choice_input = input("Modell per Nummer auswählen: ")
            model_choice = int(model_choice_input) - 1
            if 0 <= model_choice < len(models_in_brand):
                selected_model = models_in_brand[model_choice]
                print(f"Sie haben ausgewählt: {selected_model}")
            else:
                print("Ungültige Auswahl. Bitte geben Sie eine gültige Nummer ein.")
        except ValueError:
            print("Ungültige Eingabe. Bitte geben Sie eine Nummer ein.")

    # Varianten innerhalb des gewählten Modells auswählen
    variants_in_model = df[
        (df["Marke"] == selected_brand) & (df["Modell"] == selected_model)
    ]["Variante"].unique().tolist()
    print(f"\nVerfügbare Varianten für {selected_model}:")
    for i, variant in enumerate(variants_in_model):
        print(f"{i+1}. {variant}")
    selected_variant = None
    while selected_variant is None:
        try:
            variant_choice_input = input("Variante per Nummer auswählen: ")
            variant_choice = int(variant_choice_input) - 1
            if 0 <= variant_choice < len(variants_in_model):
                selected_variant = variants_in_model[variant_choice]
                print(f"Sie haben ausgewählt: {selected_variant}")
            else:
                print("Ungültige Auswahl. Bitte geben Sie eine gültige Nummer ein.")
        except ValueError:
            print("Ungültige Eingabe. Bitte geben Sie eine Nummer ein.")

    # Endgültige Auswahl der spezifischen Zeile
    selected_row = df[
        (df["Marke"] == selected_brand)
        & (df["Modell"] == selected_model)
        & (df["Variante"] == selected_variant)
    ].iloc[0]

    return {
        "initial_rv": selected_row["Neupreis_EUR"], # Anfänglicher Restwert (Neupreis)
        "age_months": selected_row["Alter_Monate"], # Alter des Fahrzeugs
        "true_residual_value_at_horizon": selected_row["Restwert_EUR"], # Tatsächlich beobachteter Restwert
        "selected_model_name": f"{selected_brand} {selected_model} {selected_variant}",
        "full_row_data": selected_row # Die vollständige Zeile für CatBoost (enthält alle 2025 Features)
    }

def choose_lease_term():
    """
    Führt Benutzer durch die Auswahl einer Vertragslaufzeit.
    """
    available_lease_terms = [12, 24, 36, 48, 60] # Verfügbare Leasing-/Finanzierungslaufzeiten in Monaten
    print("\n--- Verfügbare Vertragslaufzeiten (Monate) ---")
    for i, term in enumerate(available_lease_terms):
        print(f"{i+1}. {term} Monate")
    chosen_lease_term = None
    while chosen_lease_term is None:
        try:
            term_choice_input = input("Wählen Sie eine Laufzeit per Nummer: ")
            term_choice = int(term_choice_input) - 1
            if 0 <= term_choice < len(available_lease_terms):
                chosen_lease_term = available_lease_terms[term_choice]
                print(f"Sie haben {chosen_lease_term} Monate Laufzeit ausgewählt.")
            else:
                print("Ungültige Auswahl. Bitte geben Sie eine gültige Nummer ein.")
        except ValueError:
            print("Ungültige Eingabe. Bitte geben Sie eine Nummer ein.")
    return chosen_lease_term

def generate_simulated_test_data(csv_file, num_samples=400):
    """
    Generiert simulierte Testdaten, die einer echten Kreuzvalidierung ähneln würden.
    Für die Demonstration des t-Tests, da keine vollständige Kreuzvalidierung
    im aktuellen Workflow implementiert ist.
    """
    df_full = pd.read_csv(csv_file)

    simulated_data = []
    for _ in range(num_samples):
        # Zufällige Auswahl einer Zeile aus dem Datensatz
        random_row = df_full.sample(n=1).iloc[0]

        # Simuliere einen "tatsächlichen" Restwert für diese Simulation
        # Füge etwas Rauschen hinzu, um realistischer zu sein
        actual_rv = random_row["Restwert_EUR"] + np.random.normal(0, random_row["Neupreis_EUR"] * 0.01)
        actual_rv = max(0, actual_rv) # Restwert nicht negativ

        # Simuliere Prognose des linearen Modells (mit erwarteter Abweichung)
        linear_pred = actual_rv * (1 + np.random.normal(0.083, 0.02)) # 8.3% MAPE Abweichung

        # Simuliere Prognose des hybriden Modells (mit erwarteter Abweichung)
        hybrid_pred = actual_rv * (1 + np.random.normal(0.0167, 0.005)) # 1.67% MAPE Abweichung

        simulated_data.append({
            'actual_rv': actual_rv,
            'linear_prediction': linear_pred,
            'hybrid_prediction': hybrid_pred,
            'full_row_data': random_row # Für CatBoost Features, falls benötigt
        })
    return simulated_data


if __name__ == "__main__":
    csv_path = "hyundai_ev_restwerte.csv"
    selected_data = load_and_select_data_interactive(csv_path)
    if selected_data:
        print("\nAusgewählte Fahrzeugdaten:", selected_data)
        chosen_term = choose_lease_term()
        print("Gewählte Laufzeit:", chosen_term)
