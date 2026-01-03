# bayesian_inference.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from data_utils import load_and_select_data_interactive, choose_lease_term

def calculate_prior_parameters(selected_data):
    """
    Prior-Parameter mit 25% EV-Abwertung
    """
    initial_rv = selected_data["initial_rv"]
    age_months = selected_data["age_months"]
    observed_rv = selected_data["true_residual_value_at_horizon"]

    if age_months == 0 or initial_rv <= observed_rv:
        # Annahme: ca. 25% Wertverlust im ersten Jahr
        prior_mean_beta = initial_rv * 0.0225  # 25% jährlich = 2.10% monatlich
        prior_std_beta = initial_rv * 0.006  # 0.6% vom Neupreis als Variabilität
    else:
        # Durchschnittliche monatliche Abwertungsrate
        if age_months <= 12:
            expected_loss = initial_rv * 0.25 * (age_months/12)
        elif age_months <= 24:
            year1_loss = initial_rv * 0.25
            year2_months = age_months - 12
            year2_loss = (initial_rv - year1_loss) * 0.18 * (year2_months/12)
            expected_loss = year1_loss + year2_loss
        else:
            year1_loss = initial_rv * 0.25
            year2_loss = (initial_rv - year1_loss) * 0.18
            year3_months = age_months - 24
            year3_loss = (initial_rv - year1_loss - year2_loss) * 0.12 * (year3_months/12)
            expected_loss = year1_loss + year2_loss + year3_loss

        prior_mean_beta = expected_loss / age_months
        prior_std_beta = max(50, prior_mean_beta * 0.25)

    prior_mean_beta = max(10, prior_mean_beta)
    prior_std_beta = max(10, prior_std_beta)

    return prior_mean_beta, prior_std_beta

def likelihood(params, data_initial_rv, data_age_months, data_observed_rv, sigma_obs):
    """
    Likelihood-Funktion mit realistischen EV-Parametern
    """
    beta = params[0]

    # Vorhersage basierend auf 25% Abwertungsmodell
    if data_age_months <= 12:
        predicted_rv = data_initial_rv * (1 - 0.25 * beta * (data_age_months/12) / (data_initial_rv * 0.25/12))
    else:
        # Vereinfachte lineare Approximation für längere Zeiträume
        predicted_rv = data_initial_rv - beta * data_age_months

    predicted_rv = np.maximum(0, predicted_rv)

    return stats.norm.logpdf(data_observed_rv, loc=predicted_rv, scale=sigma_obs)

def run_bayesian_inference(selected_data):
    """
    Bayesianische Inferenz
    """
    data_initial_rv = selected_data["initial_rv"]
    data_age_months = selected_data["age_months"]
    data_observed_rv = selected_data["true_residual_value_at_horizon"]
    selected_model_name = selected_data["selected_model_name"]

    # Messrauschen für EV-Markt
    sigma_obs = 750  # Konsistent mit Kalman-Filter

    # Korrigierte Prior-Parameter
    prior_mean_beta, prior_std_beta = calculate_prior_parameters(selected_data)

    print(f"\n--- Bayesianische Inferenz ---")
    print(f"Fahrzeug: {selected_model_name}")
    print(f"Prior-Verteilung für Beta: Mittelwert = {prior_mean_beta:.2f} €/Monat, Std-Abw = {prior_std_beta:.2f} €/Monat")

    def prior(params, prior_mean_beta, prior_std_beta):
        beta = params[0]
        if beta < 0:
            return -np.inf
        return stats.norm.logpdf(beta, loc=prior_mean_beta, scale=prior_std_beta)

    def posterior(params, data_initial_rv, data_age_months, data_observed_rv, sigma_obs, prior_mean_beta, prior_std_beta):
        log_likelihood = likelihood(params, data_initial_rv, data_age_months, data_observed_rv, sigma_obs)
        log_prior = prior(params, prior_mean_beta, prior_std_beta)

        if np.isinf(log_likelihood) or np.isinf(log_prior):
            return -np.inf
        return log_likelihood + log_prior

    # Metropolis-Hastings Implementation
    def metropolis_hastings(initial_params, n_iterations=50000, proposal_std=25):
        chain = [initial_params]
        current_params = np.array(initial_params)
        current_posterior = posterior(current_params, data_initial_rv, data_age_months,
                                              data_observed_rv, sigma_obs, prior_mean_beta, prior_std_beta)

        for _ in range(n_iterations):
            proposed_params = current_params + np.random.normal(0, proposal_std, size=len(current_params))
            proposed_posterior = posterior(proposed_params, data_initial_rv, data_age_months,
                                                   data_observed_rv, sigma_obs, prior_mean_beta, prior_std_beta)

            log_alpha = proposed_posterior - current_posterior

            if log_alpha >= 0 or np.random.rand() < np.exp(log_alpha):
                current_params = proposed_params
                current_posterior = proposed_posterior

            chain.append(current_params)

        return np.array(chain)

    # MCMC-Sampling
    initial_params = [prior_mean_beta]
    mcmc_chain = metropolis_hastings(initial_params)

    # Ergebnisse analysieren
    burn_in = int(0.2 * len(mcmc_chain))
    thinned_chain = mcmc_chain[burn_in::10]
    beta_samples = thinned_chain[:, 0]

    # Visualisierung
    plt.figure(figsize=(10, 6))
    plt.hist(beta_samples, bins=50, color="skyblue", density=True, alpha=0.7, label="Korrigierte Posterior")
    plt.axvline(np.mean(beta_samples), color="red", linestyle="--", label=f"Mittelwert: {np.mean(beta_samples):.2f}")
    plt.axvline(np.percentile(beta_samples, 2.5), color="green", linestyle=":", label="2.5% Quantil")
    plt.axvline(np.percentile(beta_samples, 97.5), color="green", linestyle=":", label="97.5% Quantil")
    plt.legend()
    plt.title(f"Posterior-Verteilung \n{selected_model_name}")
    plt.xlabel(r"Monatliche Wertminderungsrate $\beta$ (€/Monat)")
    plt.ylabel("Dichte")
    plt.grid()
    plt.show()

    print(f"\n--- Bayesianische Inferenz Ergebnisse ---")
    print(f"Fahrzeug: {selected_model_name}")
    print(f"Posterior-Mittelwert: {np.mean(beta_samples):.2f} €/Monat")
    print(f"95% Glaubwürdigkeitsintervall: [{np.percentile(beta_samples, 2.5):.2f}, {np.percentile(beta_samples, 97.5):.2f}] €/Monat")

    return {
        "mean_beta_posterior": np.mean(beta_samples),
        "std_beta_posterior": np.std(beta_samples),
        "credible_interval_beta": [np.percentile(beta_samples, 2.5), np.percentile(beta_samples, 97.5)]
    }

if __name__ == "__main__":
    csv_path = "hyundai_ev_restwerte.csv"
    selected_data = load_and_select_data_interactive(csv_path)
    if selected_data:
        bayesian_output = run_bayesian_inference(selected_data)
