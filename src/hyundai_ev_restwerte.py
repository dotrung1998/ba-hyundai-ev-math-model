# hyundai_ev_restwerte.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_realistic_ev_data_2025(num_samples=2000):
    """
    DE: Generiert realistische EV-Restwertdaten basierend auf 2025 Marktdaten
        mit aktualisierten Merkmalen: Karosserieform, Spitzenleistung, WLTP-Verbrauch,
        WLTP-Reichweite und Elektropreis-Marktfaktoren
    EN: Generates realistic EV residual value data based on 2025 market data
        with updated features: Body style, Peak power, WLTP consumption,
        WLTP range, and Electricity price market factors
    """

    # DE: Aktualisierte Hyundai EV-Modelle mit 2025 Spezifikationen
    # EN: Updated Hyundai EV models with 2025 specifications
    modelle_details = {
        "Hyundai": {
            "KONA Elektro": {
                "Varianten": ["Standard Range 49kWh", "Long Range 65kWh"],
                "Neupreise": [35990, 42990],
                "Batterien": [49, 65],
                "Spitzenleistung_kW": [99, 150], 
                "WLTP_Verbrauch_kWh_100km": [13.8, 13.4],
                "WLTP_Reichweite_km": [320, 480],
                "Karosserieform": "SUV",
                "Wertverlust_Faktor": 0.28
            },
            "IONIQ 5": {
                "Varianten": ["RWD 58kWh", "RWD 84kWh", "AWD 84kWh"],
                "Neupreise": [47900, 52900, 57900],
                "Batterien": [58, 84, 84],
                "Spitzenleistung_kW": [125, 168, 239],
                "WLTP_Verbrauch_kWh_100km": [16.8, 17.1, 18.2],
                "WLTP_Reichweite_km": [345, 490, 460],
                "Karosserieform": "SUV",
                "Wertverlust_Faktor": 0.248
            },
            "IONIQ 5 N": {
                "Varianten": ["Performance 84kWh"],
                "Neupreise": [74900],
                "Batterien": [84],
                "Spitzenleistung_kW": [448],
                "WLTP_Verbrauch_kWh_100km": [22.5],
                "WLTP_Reichweite_km": [355],
                "Karosserieform": "SUV",
                "Wertverlust_Faktor": 0.246
            },
            "IONIQ 6": {
                "Varianten": ["Standard Range 53kWh", "Long Range 77kWh"],
                "Neupreise": [49900, 54900],
                "Batterien": [53, 77],
                "Spitzenleistung_kW": [111, 168],
                "WLTP_Verbrauch_kWh_100km": [13.9, 14.3],
                "WLTP_Reichweite_km": [385, 540],
                "Karosserieform": "Limousine",
                "Wertverlust_Faktor": 0.26
            },
        }
    }

    farben = ["Schwarz", "Weiß", "Blau", "Rot", "Grau", "Silber", "Grün", "Orange"]
    regionen = ["DE-Nord", "DE-Süd", "DE-Ost", "DE-West", "DE-Mitte"]

    # DE: Marktfaktoren für Elektropreis basierend auf 2025 Prognosen
    # EN: Market factors for electricity price based on 2025 forecasts
    marktfaktoren_elektropreis = {
        "niedrig": 1.05,    # DE: Niedrige Strompreise begünstigen EV-Nachfrage / EN: Low electricity prices favor EV demand
        "mittel": 1.0,      # Baseline
        "hoch": 0.96        # DE: Hohe Strompreise reduzieren EV-Attraktivität / EN: High electricity prices reduce EV attractiveness
    }
    marktfaktoren_nachfrage = {"schwach": 0.90, "mittel": 1.0, "stark": 1.08}

    def simuliere_realistischen_restwert_2025(neupreis, alter_monate, laufleistung_km,
                                             spitzenleistung_kw, wltp_verbrauch, wltp_reichweite,
                                             karosserieform, zustand, elektropreis_faktor,
                                             nachfrage_faktor, wertverlust_faktor):
        """
        DE: Erweiterte Restwertberechnung mit neuen technischen Merkmalen
        EN: Extended residual value calculation with new technical features
        """

        # DE: Basis-Wertverlustkurve
        # EN: Base depreciation curve
        alter_jahre = alter_monate / 12.0

        if alter_jahre <= 1:
            basis_wertverlust = 0.25 * alter_jahre
        elif alter_jahre <= 2:
            basis_wertverlust = 0.25 + 0.15 * (alter_jahre - 1)
        elif alter_jahre <= 3:
            basis_wertverlust = 0.40 + 0.10 * (alter_jahre - 2)
        else:
            basis_wertverlust = 0.50 + 0.05 * (alter_jahre - 3)

        basis_wertverlust = basis_wertverlust * wertverlust_faktor * 2
        simulierter_wert = neupreis * (1 - basis_wertverlust)

        # DE: Laufleistungseffekt (EV-spezifisch: 12.215 km/Jahr Durchschnitt)
        # EN: Mileage effect (EV-specific: 12,215 km/year average)
        durchschnitt_km_pro_jahr = 12215
        erwartete_laufleistung = (alter_monate / 12) * durchschnitt_km_pro_jahr
        laufleistung_abweichung = laufleistung_km - erwartete_laufleistung

        if laufleistung_abweichung > 0:
            simulierter_wert -= laufleistung_abweichung * 0.08
        else:
            simulierter_wert += min(abs(laufleistung_abweichung) * 0.04, neupreis * 0.05)

        # DE: Spitzenleistung: Bonus für hohe Leistung
        # EN: Peak power: Bonus for high performance
        if spitzenleistung_kw > 200:
            simulierter_wert += 2500  # DE: Performance-Bonus / EN: Performance Bonus
        elif spitzenleistung_kw > 150:
            simulierter_wert += 1500
        elif spitzenleistung_kw < 100:
            simulierter_wert -= 1000  # DE: Abzug für niedrige Leistung / EN: Deduction for low power

        # DE: WLTP Energieverbrauch: Effizienz-Bonus/Malus
        # EN: WLTP Energy consumption: Efficiency Bonus/Malus
        if wltp_verbrauch < 14.0:  # DE: Sehr effizient / EN: Very efficient
            simulierter_wert += 1800
        elif wltp_verbrauch < 16.0:  # DE: Durchschnittlich effizient / EN: Average efficiency
            simulierter_wert += 800
        elif wltp_verbrauch > 20.0:  # DE: Ineffizient / EN: Inefficient
            simulierter_wert -= 1200

        # DE: WLTP Reichweite: Reichweiten-Premium
        # EN: WLTP Range: Range Premium
        if wltp_reichweite > 500:  # DE: Über 500km = Premium / EN: Over 500km = Premium
            simulierter_wert += 2200
        elif wltp_reichweite > 400:  # DE: Über 400km = gut / EN: Over 400km = good
            simulierter_wert += 1200
        elif wltp_reichweite < 300:  # DE: Unter 300km = Abzug / EN: Under 300km = deduction
            simulierter_wert -= 1500

        # DE: Karosserieform-Faktoren
        # EN: Body style factors
        karosserieform_faktoren = {
            "SUV": 1.08,        # DE: SUVs sind beliebt / EN: SUVs are popular
            "Limousine": 1.02,  # DE: Limousinen stabil / EN: Sedans stable
            "Kleinwagen": 0.95  # DE: Kleinwagen weniger wertstabil / EN: Small cars hold less value
        }
        simulierter_wert *= karosserieform_faktoren.get(karosserieform, 1.0)

        # DE: Zustand: Kritischer Faktor
        # EN: Condition: Critical factor
        zustand_multiplikator = {
            1: 0.65, 2: 0.78, 3: 0.92, 4: 1.05, 5: 1.12
        }
        simulierter_wert *= zustand_multiplikator.get(zustand, 0.92)

        # DE: Marktfaktoren
        # EN: Market factors
        simulierter_wert *= elektropreis_faktor
        simulierter_wert *= nachfrage_faktor

        # DE: Marktschwankung
        # EN: Market fluctuation
        marktschwankung = np.random.normal(0, neupreis * 0.02)
        simulierter_wert += marktschwankung

        # DE: Plausibilitätsgrenzen
        # EN: Plausibility limits
        simulierter_wert = max(neupreis * 0.15, min(simulierter_wert, neupreis * 0.95))

        return round(simulierter_wert, 0)

    simulierte_daten = []
    fahrzeug_id_counter = 1

    for _ in range(num_samples):
        marke = random.choice(list(modelle_details.keys()))
        model_name = random.choice(list(modelle_details[marke].keys()))
        model_info = modelle_details[marke][model_name]

        variant_idx = random.randrange(len(model_info["Varianten"]))
        variant_name = model_info["Varianten"][variant_idx]
        neupreis = model_info["Neupreise"][variant_idx]
        batterie_kwh = model_info["Batterien"][variant_idx]
        spitzenleistung_kw = model_info["Spitzenleistung_kW"][variant_idx]
        wltp_verbrauch = model_info["WLTP_Verbrauch_kWh_100km"][variant_idx]
        wltp_reichweite = model_info["WLTP_Reichweite_km"][variant_idx]
        karosserieform = model_info["Karosserieform"]
        wertverlust_faktor = model_info["Wertverlust_Faktor"]

        # DE: Realistische Altersverteilung
        # EN: Realistic age distribution
        alter_gewichte = [40, 30, 20, 10]
        alter_bereiche = [(1, 12), (13, 24), (25, 36), (37, 60)]
        alter_bereich = random.choices(alter_bereiche, weights=alter_gewichte)[0]
        alter = random.randint(*alter_bereich)

        # DE: Laufleistung
        # EN: Mileage
        basis_laufleistung = alter * (12215 / 12)
        laufleistung_variation = np.random.normal(0, basis_laufleistung * 0.3)
        laufleistung = max(500, int(basis_laufleistung + laufleistung_variation))

        farbe_val = random.choice(farben)
        region_val = random.choice(regionen)
        zustand_val = random.choices([2, 3, 4, 5], weights=[5, 25, 45, 25])[0]

        elektropreis_key = random.choices(
            list(marktfaktoren_elektropreis.keys()),
            weights=[20, 60, 20]
        )[0]
        nachfrage_key = random.choices(
            list(marktfaktoren_nachfrage.keys()),
            weights=[25, 50, 25]
        )[0]

        elektropreis_faktor = marktfaktoren_elektropreis[elektropreis_key]
        nachfrage_faktor = marktfaktoren_nachfrage[nachfrage_key]

        historischer_preis = simuliere_realistischen_restwert_2025(
            neupreis, alter, laufleistung, spitzenleistung_kw, wltp_verbrauch,
            wltp_reichweite, karosserieform, zustand_val, elektropreis_faktor,
            nachfrage_faktor, wertverlust_faktor
        )

        wertverlust_prozent = round(((neupreis - historischer_preis) / neupreis) * 100, 1)

        # DE: Verkaufsdatum
        # EN: Date of sale
        tage_gewichte = [50, 30, 15, 5]
        tage_bereiche = [(1, 90), (91, 365), (366, 730), (731, 1095)]
        tage_bereich = random.choices(tage_bereiche, weights=tage_gewichte)[0]
        tage_zurueck = random.randint(*tage_bereich)
        verkaufsdatum = (datetime.now() - timedelta(days=tage_zurueck)).strftime("%Y-%m-%d")

        simulierte_daten.append([
            fahrzeug_id_counter, marke, model_name, variant_name, neupreis,
            alter, laufleistung, batterie_kwh, karosserieform, spitzenleistung_kw,
            wltp_verbrauch, wltp_reichweite, farbe_val, region_val, zustand_val,
            historischer_preis, wertverlust_prozent, verkaufsdatum,
            elektropreis_key, nachfrage_key
        ])

        fahrzeug_id_counter += 1

    df_simuliert = pd.DataFrame(simulierte_daten, columns=[
        "FahrzeugID", "Marke", "Modell", "Variante", "Neupreis_EUR",
        "Alter_Monate", "Laufleistung_km", "Batteriegroesse_kWh",
        "Karosserieform", "Spitzenleistung_kW", "WLTP_Energieverbrauch_kWh_100km",
        "WLTP_Elektrische_Reichweite_km", "Farbe", "Region",
        "Zustand_Skala_1_5", "Restwert_EUR", "Wertverlust_Prozent",
        "Datum_Verkauf", "Marktfaktor_Elektropreis", "Marktfaktor_EVNachfrage"
    ])

    return df_simuliert

def print_dataset_statistics_2025(df):
    """
    DE: Druckt erweiterte Statistiken für den 2025 Datensatz
    EN: Prints extended statistics for the 2025 dataset
    """
    print("\n=== DATASET STATISTIKEN 2025 ===")
    print(f"Gesamtanzahl Datensätze: {len(df)}")
    print(f"Durchschnittlicher Wertverlust: {df['Wertverlust_Prozent'].mean():.1f}%")
    print(f"Durchschnittliches Alter: {df['Alter_Monate'].mean():.1f} Monate")
    print(f"Durchschnittliche Laufleistung: {df['Laufleistung_km'].mean():,.0f} km")
    print(f"Durchschnittliche Spitzenleistung: {df['Spitzenleistung_kW'].mean():.0f} kW")
    print(f"Durchschnittlicher WLTP-Verbrauch: {df['WLTP_Energieverbrauch_kWh_100km'].mean():.1f} kWh/100km")
    print(f"Durchschnittliche WLTP-Reichweite: {df['WLTP_Elektrische_Reichweite_km'].mean():.0f} km")

    print("\n=== KAROSSERIEFORM-VERTEILUNG ===")
    print(df['Karosserieform'].value_counts())

    print("\n=== WERTVERLUST NACH KAROSSERIEFORM ===")
    wertverlust_nach_form = df.groupby('Karosserieform')['Wertverlust_Prozent'].mean().sort_values()
    for form, wertverlust in wertverlust_nach_form.items():
        print(f"{form}: {wertverlust:.1f}%")

    print("\n=== LEISTUNGSKLASSEN ===")
    df['Leistungsklasse'] = pd.cut(df['Spitzenleistung_kW'],
                                   bins=[0, 100, 150, 200, 500],
                                   labels=['Einsteiger (<100kW)', 'Mittelklasse (100-150kW)',
                                          'Oberklasse (150-200kW)', 'Performance (>200kW)'])
    print(df['Leistungsklasse'].value_counts())

if __name__ == "__main__":
    print("Generiere realistische EV-Restwertdaten 2025 mit aktualisierten Merkmalen...")

    # DE: Datensatz generieren
    # EN: Generate dataset
    simulated_df = generate_realistic_ev_data_2025(num_samples=2000)

    # DE: CSV speichern
    # EN: Save CSV
    csv_filename = "hyundai_ev_restwerte.csv"
    simulated_df.to_csv(csv_filename, index=False, encoding='utf-8')

    print(f"\n CSV-Datei '{csv_filename}' erfolgreich erstellt!")

    # DE: Erweiterte Statistiken anzeigen
    # EN: Show extended statistics
    print_dataset_statistics_2025(simulated_df)

    # DE: Beispieldaten anzeigen
    # EN: Show sample data
    print("\n=== BEISPIELDATEN (erste 3 Zeilen) ===")
    print(simulated_df.head(3).to_string())

    print(f"\n Der aktualisierte Datensatz ist bereit für Machine Learning Modelle!")
    print(f" Datei gespeichert als: {csv_filename}")