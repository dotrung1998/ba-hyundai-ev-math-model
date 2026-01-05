# kalman_filter.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_and_select_data_interactive, choose_lease_term

def run_kalman_filter(selected_data, target_forecast_months):
    """
    DE: Kalman-Filter mit Neusser (2016) Notation und Visualisierung der realistischen Abwertungsrate.
    EN: Kalman Filter using Neusser (2016) notation and visualization of realistic depreciation rate.
    """
    # DE: Extrahiere Daten
    # EN: Extract data
    initial_rv = selected_data["initial_rv"]
    age_months_observed = selected_data["age_months"]
    true_residual_value_at_horizon = selected_data["true_residual_value_at_horizon"]
    selected_model_name = selected_data["selected_model_name"]

    print(f"\n--- Kalman-Filter mit Neusser (2016) Notation ---")
    print(f"Fahrzeug: {selected_model_name}")

    # DE: Zeitvektor
    # EN: Time vector
    dt = 1
    time_vector = np.arange(0, target_forecast_months + dt, dt)

    # --- ANPASSUNG START ---
    # DE: Berechne eine zeitlich veränderliche "realistische" Abwertungsrate als Stufenfunktion
    # EN: Calculate a time-varying "realistic" depreciation rate as a step function
    realistic_monthly_rates = []
    value_after_year1 = initial_rv * (1 - 0.25)
    value_after_year2 = value_after_year1 * (1 - 0.18)

    for t in time_vector:
        if t <= 12:
            rate = (initial_rv * 0.25) / 12
        elif t <= 24:
            rate = (value_after_year1 * 0.18) / 12
        else:
            rate = (value_after_year2 * 0.12) / 12
        realistic_monthly_rates.append(rate)

    realistic_monthly_rates = np.array(realistic_monthly_rates)
    # DE: Die alte, konstante `monthly_rate`-Berechnung wurde entfernt.
    # EN: The old, constant `monthly_rate` calculation has been removed.
    # --- ANPASSUNG ENDE ---

    # DE: Erstelle die realistische Wertkurve für die Visualisierung
    # EN: Create the realistic value curve for visualization
    true_rv_curve_plotting = []
    for t in time_vector:
        if t <= 12:
            rv = initial_rv * (1 - 0.25 * (t / 12))
        elif t <= 24:
            # DE: Berechnung basierend auf dem Restwert nach Jahr 1
            # EN: Calculation based on residual value after year 1
            rv = value_after_year1 * (1 - 0.18 * ((t - 12) / 12))
        else:
            # DE: Berechnung basierend auf dem Restwert nach Jahr 2
            # EN: Calculation based on residual value after year 2
            rv = value_after_year2 * (1 - 0.12 * ((t - 24) / 12))
        true_rv_curve_plotting.append(max(0, rv))
    true_rv_curve_plotting = np.array(true_rv_curve_plotting)

    # DE: Systemmatrizen nach Neusser (2016)
    # EN: System matrices according to Neusser (2016)
    F = np.array([[1, -dt], [0, 1]])
    A = np.array([[0]])
    G = np.array([[1, 0]])

    # DE: Rauschparameter für 2025 EV-Markt
    # EN: Noise parameters for 2025 EV market
    Q = np.array([[150, 0], [0, 15]])
    R = np.array([[750 ** 2]])

    # DE: Simuliere verrauschte Beobachtungen
    # EN: Simulate noisy observations
    observed_market_prices = true_rv_curve_plotting + np.random.normal(0, 750, len(time_vector))
    if age_months_observed > 0 and age_months_observed < len(time_vector):
        idx_at_observed = int(age_months_observed / dt)
        observed_market_prices[idx_at_observed] = true_residual_value_at_horizon + np.random.normal(0, 750 * 0.2)

    # DE: Initialisierung
    # EN: Initialization
    # DE: Verwende die Rate am Anfang des Zeitraums für die Initialisierung
    # EN: Use the rate at the beginning of the period for initialization
    initial_estimated_rate = realistic_monthly_rates[0]
    X_hat = np.array([[initial_rv], [initial_estimated_rate]])
    P = np.array([[1800**2, 0], [0, 90**2]])

    # DE: Speicher für Ergebnisse
    # EN: Storage for results
    estimated_states = []
    confidence_intervals_rv = []
    estimated_rates = []

    print(f"\n--- Kalman-Filter-Iteration mit Neusser-Notation ---")
    # DE: Kalman-Filter Iteration
    # EN: Kalman Filter Iteration
    for k in range(len(time_vector)):
        # DE: Vorhersageschritt
        # EN: Prediction step
        X_hat_minus = F @ X_hat
        P_minus = F @ P @ F.T + Q

        # DE: Aktualisierungsschritt
        # EN: Update step
        if k < len(observed_market_prices):
            Y_k = observed_market_prices[k]
            nu = Y_k - A[0] - G @ X_hat_minus
            S = G @ P_minus @ G.T + R
            K = P_minus @ G.T @ np.linalg.inv(S)
            X_hat = X_hat_minus + K @ nu
            P = (np.eye(2) - K @ G) @ P_minus
        else:
            X_hat = X_hat_minus
            P = P_minus

        estimated_states.append(X_hat[0, 0])
        confidence_intervals_rv.append(1.96 * np.sqrt(P[0, 0]))
        estimated_rates.append(X_hat[1, 0])

    estimated_states = np.array(estimated_states)
    confidence_intervals_rv = np.array(confidence_intervals_rv)
    estimated_rates = np.array(estimated_rates)

    # DE: Plotting der Ergebnisse
    # EN: Plotting the results
    plt.figure(figsize=(12, 10))

    # DE: Plot 1: Restwertprognose
    # EN: Plot 1: Residual Value Forecast
    plt.subplot(2, 1, 1)
    plt.plot(time_vector, true_rv_curve_plotting, 'k-', linewidth=2, label="Realistische EV-Wertkurve")
    plt.plot(time_vector, observed_market_prices, 'bo', markersize=4, label="Beobachtete Marktpreise")
    plt.plot(time_vector, estimated_states, 'r--', linewidth=2, label="Kalman-Filter Schätzung")
    plt.fill_between(
        time_vector,
        estimated_states - confidence_intervals_rv,
        estimated_states + confidence_intervals_rv,
        color='red', alpha=0.2, label="95% Konfidenzintervall")
    if age_months_observed <= target_forecast_months:
        plt.plot(age_months_observed, true_residual_value_at_horizon, 'gx',
                 markersize=10, mew=2, label=f"Tatsächlicher Wert bei {age_months_observed} Monaten")
    plt.title(f"EV-Restwertprognose\n{selected_model_name} (Laufzeit {target_forecast_months} Monate)")
    plt.xlabel("Zeit (Monate)")
    plt.ylabel("Restwert (€)")
    plt.legend()
    plt.grid()

    # DE: Plot 2: Abwertungsrate
    # EN: Plot 2: Depreciation Rate
    plt.subplot(2, 1, 2)
    estimated_rates[estimated_rates < 0] = 0

    plt.plot(time_vector, realistic_monthly_rates, 'g-', linewidth=2, label="Realistische EV-Abwertungsrate")
    plt.plot(time_vector, estimated_rates, 'm--', linewidth=2, label="Geschätzte Abwertungsrate")
    plt.title("Monatliche EV-Abwertungsrate")
    plt.xlabel("Zeit (Monate)")
    plt.ylabel("Rate (€/Monat)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # DE: Finale Ergebnisse
    # EN: Final results
    final_predicted_rv = estimated_states[-1]
    wertverlust_prozent = ((initial_rv - final_predicted_rv) / initial_rv) * 100

    print(f"\n--- Kalman-Filter Ergebnisse ---")
    print(f"Fahrzeug: {selected_model_name}")
    print(f"Laufzeit: {target_forecast_months} Monate")
    print(f"Neupreis: {initial_rv:.2f} €")
    print(f"Prognostizierter Restwert: {final_predicted_rv:.2f} €")
    print(f"Wertverlust: {wertverlust_prozent:.1f}% (realistische EV-Abwertung)")
    print(f"Geschätzte monatliche Abwertungsrate: {estimated_rates[-1]:.2f} €/Monat")
    print(f"95% Konfidenzintervall: +/- {confidence_intervals_rv[-1]:.2f} €")

    return {
        "ssm_baseline_rv_prediction": final_predicted_rv,
        "ssm_baseline_rv_curve": estimated_states,
        "forecast_months": target_forecast_months,
        "initial_rv": initial_rv,
        "selected_model_name": selected_model_name,
        "estimated_rv_at_age_observed": estimated_states[time_vector == age_months_observed][0] if age_months_observed <= target_forecast_months else None,
        "age_months_observed": age_months_observed,
        "true_residual_value_at_horizon": true_residual_value_at_horizon,
        "wertverlust_prozent": wertverlust_prozent,
        "confidence_interval": confidence_intervals_rv[-1]
    }