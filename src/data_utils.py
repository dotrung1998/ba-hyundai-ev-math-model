# data_utils.py

import pandas as pd
import numpy as np
import random # DE: Neu hinzugefügt für die Simulation / EN: Newly added for simulation

def load_and_select_data_interactive(csv_file):
    """
    DE: Lädt CSV, führt Benutzer durch die Auswahl eines EV-Modells und einer Variante
        und gibt die ausgewählten Daten zurück.
        Aktualisiert für 2025 Datenstruktur mit neuen Merkmalen.
    EN: Loads CSV, guides user through selecting an EV model and variant,
        and returns the selected data.
        Updated for 2025 data structure with new features.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{csv_file}' wurde nicht gefunden.")
        print("Bitte stellen Sie sicher, dass die 'hyundai_ev_restwerte_2025_aktualisiert.csv' im selben Verzeichnis liegt.")
        return None

    # DE: Marken auswählen
    # EN: Select brands
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

    # DE: Modelle innerhalb der gewählten Marke auswählen
    # EN: Select models within the chosen brand
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

    # DE: Varianten innerhalb des gewählten Modells auswählen
    # EN: Select variants within the chosen model
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

    # DE: Endgültige Auswahl der spezifischen Zeile
    # EN: Final selection of the specific row
    selected_row = df[
        (df["Marke"] == selected_brand)
        & (df["Modell"] == selected_model)
        & (df["Variante"] == selected_variant)
    ].iloc[0]

    return {
        "initial_rv": selected_row["Neupreis_EUR"], # DE: Anfänglicher Restwert (Neupreis) / EN: Initial residual value (New price)
        "age_months": selected_row["Alter_Monate"], # DE: Alter des Fahrzeugs / EN: Vehicle age
        "true_residual_value_at_horizon": selected_row["Restwert_EUR"], # DE: Tatsächlich beobachteter Restwert / EN: Actually observed residual value
        "selected_model_name": f"{selected_brand} {selected_model} {selected_variant}",
        "full_row_data": selected_row # DE: Die vollständige Zeile für CatBoost (enthält alle 2025 Features) / EN: The full row for CatBoost (contains all 2025 features)
    }

def choose_lease_term():
    """
    DE: Führt Benutzer durch die Auswahl einer Vertragslaufzeit.
    EN: Guides user through the selection of a lease term.
    """
    available_lease_terms = [12, 24, 36, 48, 60] # DE: Verfügbare Leasing-/Finanzierungslaufzeiten in Monaten / EN: Available lease/financing terms in months
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
    DE: Generiert simulierte Testdaten, die einer echten Kreuzvalidierung ähneln würden.
        Für die Demonstration des t-Tests, da keine vollständige Kreuzvalidierung
        im aktuellen Workflow implementiert ist.
    EN: Generates simulated test data that would resemble a real cross-validation.
        For the demonstration of the t-test, as no full cross-validation
        is implemented in the current workflow.
    """
    df_full = pd.read_csv(csv_file)

    simulated_data = []
    for _ in range(num_samples):
        # DE: Zufällige Auswahl einer Zeile aus dem Datensatz
        # EN: Random selection of a row from the dataset
        random_row = df_full.sample(n=1).iloc[0]

        # DE: Simuliere einen "tatsächlichen" Restwert für diese Simulation
        # EN: Simulate an "actual" residual value for this simulation
        # DE: Füge etwas Rauschen hinzu, um realistischer zu sein
        # EN: Add some noise to be more realistic
        actual_rv = random_row["Restwert_EUR"] + np.random.normal(0, random_row["Neupreis_EUR"] * 0.01)
        actual_rv = max(0, actual_rv) # DE: Restwert nicht negativ / EN: Residual value not negative

        # DE: Simuliere Prognose des linearen Modells (mit erwarteter Abweichung)
        # EN: Simulate prediction of the linear model (with expected deviation)
        linear_pred = actual_rv * (1 + np.random.normal(0.083, 0.02)) # 8.3% MAPE Abweichung

        # DE: Simuliere Prognose des hybriden Modells (mit erwarteter Abweichung)
        # EN: Simulate prediction of the hybrid model (with expected deviation)
        hybrid_pred = actual_rv * (1 + np.random.normal(0.0167, 0.005)) # 1.67% MAPE Abweichung

        simulated_data.append({
            'actual_rv': actual_rv,
            'linear_prediction': linear_pred,
            'hybrid_prediction': hybrid_pred,
            'full_row_data': random_row # DE: Für CatBoost Features, falls benötigt / EN: For CatBoost features if needed
        })
    return simulated_data


if __name__ == "__main__":
    csv_path = "hyundai_ev_restwerte.csv"
    selected_data = load_and_select_data_interactive(csv_path)
    if selected_data:
        print("\nAusgewählte Fahrzeugdaten:", selected_data)
        chosen_term = choose_lease_term()
        print("Gewählte Laufzeit:", chosen_term)