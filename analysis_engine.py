# ==============================================================================
# PARKINSON'S SYMPTOM PROGRESSION ANALYZER - ANALYSIS ENGINE (CORRECTED)
# Author: Onessa Crispeyn
#
# Description:
# This version encapsulates the entire analysis pipeline within a
# reusable `ParkinsonsAnalyzer` class. It is designed to be imported and used
# by a web server or any other application.
# ==============================================================================

import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-GUI environments like servers
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# Centralized configuration for easy tweaking of model parameters
CONFIG = {
    "data_path": './parkinsons_updrs.data',
    "interpolation_points": 100,
    "n_clusters": 2,
    "k_matches": 10,
    "age_window": 5,
    "random_state": 42
}

class ParkinsonsAnalyzer:
    """A class to encapsulate the entire Parkinson's analysis pipeline."""

    def __init__(self, config):
        """
        Initializes the analyzer by loading data, training models, and preparing assets.
        This heavy lifting is done only once when the class is instantiated.
        """
        print("Initializing Analyzer...")
        self.config = config
        self.df = self._load_and_preprocess_data(config["data_path"])
        self.all_trajectories = self._standardize_all_trajectories(self.df)
        
        # Train the cohort discovery model
        self._discover_cohorts()
        
        # Prepare a global scaler for consistent scaling of new data
        self.global_scaler = StandardScaler().fit(self.df[['total_UPDRS']])
        print("✅ Analyzer is ready.")

    def _load_and_preprocess_data(self, path):
        """Loads and cleans the raw Parkinson's data."""
        df = pd.read_csv(path)
        df = df[['subject#', 'age', 'sex', 'test_time', 'total_UPDRS']]
        df[['total_UPDRS', 'test_time', 'age']] = df[['total_UPDRS', 'test_time', 'age']].apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        return df

    def _standardize_all_trajectories(self, df):
        """Interpolates all patient trajectories to a standard length."""
        def normalize_and_interpolate(patient_df):
            if len(patient_df) < 2: return None
            min_time, max_time = patient_df['test_time'].min(), patient_df['test_time'].max()
            if max_time == min_time: return None
            
            # Normalize time from 0 to 1 to handle different observation periods
            time_normalized = (patient_df['test_time'] - min_time) / (max_time - min_time)
            
            interp_func = interp1d(time_normalized, patient_df['total_UPDRS'], kind='linear', fill_value="extrapolate")
            return interp_func(np.linspace(0, 1, self.config["interpolation_points"]))

        all_trajectories = {sid: normalize_and_interpolate(pdf) for sid, pdf in df.groupby('subject#')}
        return {sid: traj for sid, traj in all_trajectories.items() if traj is not None and not np.isnan(traj).any()}

    def _discover_cohorts(self):
        """Uses K-Means clustering to discover and label patient cohorts."""
        subject_ids = list(self.all_trajectories.keys())
        trajectory_matrix = np.array(list(self.all_trajectories.values()))
        
        scaler = StandardScaler()
        scaled_trajectories = scaler.fit_transform(trajectory_matrix)
        
        kmeans = KMeans(n_clusters=self.config["n_clusters"], random_state=self.config["random_state"], n_init=10)
        clusters = kmeans.fit_predict(scaled_trajectories)
        
        cohort_df = pd.DataFrame({'subject#': subject_ids, 'cluster': clusters})
        
        # Calculate progression rate to label clusters automatically
        rates_df = self.df.groupby('subject#').apply(
            lambda x: (x['total_UPDRS'].max() - x['total_UPDRS'].min()) / (x['test_time'].max() - x['test_time'].min()) if (x['test_time'].max() - x['test_time'].min()) > 0 else 0
        ).reset_index(name='rate')
        
        cohort_df = pd.merge(cohort_df, rates_df, on='subject#')
        
        cluster_avg_rates = cohort_df.groupby('cluster')['rate'].mean()
        fast_cluster_id = cluster_avg_rates.idxmax()
        cohort_df['cohort'] = np.where(cohort_df['cluster'] == fast_cluster_id, 'Fast Progressor', 'Slow Progressor')
        
        # Merge cohort info back into the main dataframe
        self.df = pd.merge(self.df, cohort_df[['subject#', 'cohort']], on='subject#')

    def _find_top_k_matches(self, new_patient_series, patient_age, patient_sex):
        """Finds top matches by comparing the input series to historical starting segments."""
        # The input series is already interpolated to 100 points
        input_series_scaled = self.global_scaler.transform(np.array(new_patient_series).reshape(-1, 1))
        
        # Filter candidates by age and sex if provided
        target_df = self.df
        if patient_age is not None and patient_sex is not None:
             filtered_df = self.df[
                self.df['age'].between(patient_age - self.config["age_window"], patient_age + self.config["age_window"]) &
                (self.df['sex'] == patient_sex)
            ]
             # Fallback if filters are too restrictive
             if len(filtered_df['subject#'].unique()) >= self.config["k_matches"]:
                 target_df = filtered_df
        
        all_matches = []
        for sid in target_df['subject#'].unique():
            if sid in self.all_trajectories:
                historical_traj = self.all_trajectories[sid]
                # Scale the historical trajectory for a fair comparison
                historical_traj_scaled = self.global_scaler.transform(historical_traj.reshape(-1, 1))
                
                # DTW compares the entire shape of the time-normalized series
                dist = dtw.distance(input_series_scaled.flatten(), historical_traj_scaled.flatten(), use_pruning=True)
                
                all_matches.append({
                    "match_id": sid, "distance": dist,
                    "cohort": target_df[target_df['subject#'] == sid]['cohort'].iloc[0],
                    "full_trajectory": historical_traj
                })
        
        if not all_matches:
            return []
            
        return sorted(all_matches, key=lambda x: x['distance'])[:self.config["k_matches"]]

    def _generate_weighted_forecast(self, top_matches, raw_input_len):
        """Generates a forecast from a weighted average of the top matches."""
        distances = np.array([match['distance'] for match in top_matches])
        
        # Get the "future" part of each matched trajectory
        future_trajectories = np.array([match['full_trajectory'][raw_input_len:] for match in top_matches])
        
        # Inverse distance weighting
        weights = 1 / (distances + 1e-9)
        normalized_weights = weights / np.sum(weights)
        
        # Calculate the weighted average forecast
        forecast = np.average(future_trajectories, axis=0, weights=normalized_weights)
        
        # Calculate predicted cohort and confidence based on weights
        cohort_weights = {}
        for match, weight in zip(top_matches, normalized_weights):
            cohort_weights[match['cohort']] = cohort_weights.get(match['cohort'], 0) + weight
        
        predicted_cohort = max(cohort_weights, key=cohort_weights.get)
        confidence = (cohort_weights[predicted_cohort] / sum(cohort_weights.values())) * 100
        
        return forecast, predicted_cohort, confidence

    def predict(self, new_patient_series, patient_age, patient_sex):
        """The main public method. Takes new patient data and returns a full analysis."""
        if len(new_patient_series) < 2:
            return {"error": "Input series must have at least 2 data points."}
        
        raw_input_len = len(new_patient_series)
            
        # Interpolate the user's data to the standard 100 points for shape comparison
        interp_func = interp1d(np.linspace(0, 1, raw_input_len), new_patient_series, kind='linear', fill_value="extrapolate")
        input_interp = interp_func(np.linspace(0, 1, self.config["interpolation_points"]))

        top_matches = self._find_top_k_matches(input_interp, patient_age, patient_sex)
        
        if not top_matches:
            return {"error": "Could not find any suitable matches for the given patient context."}
        
        forecast, cohort, confidence = self._generate_weighted_forecast(top_matches, raw_input_len)
        
        return {
            "input_data": new_patient_series,
            "forecast": forecast,
            "predicted_cohort": cohort,
            "confidence": confidence,
            "best_match": top_matches[0],
            "top_matches_count": len(top_matches)
        }

    def generate_dtw_plot(self, input_series, match_series):
        """Generates the DTW path visualization for XAI."""
        # To make the plot interpretable, we compare the interpolated user data
        # with the full trajectory of the best match.
        x_orig_input = np.linspace(0, 1, len(input_series))
        f_input = interp1d(x_orig_input, input_series, kind='linear')
        s1 = f_input(np.linspace(0, 1, self.config["interpolation_points"]))

        # The match series is already interpolated
        s2 = np.array(match_series)

        # We must calculate the path on the SCALED data, as this is what the model used
        s1_scaled = self.global_scaler.transform(s1.reshape(-1, 1)).flatten()
        s2_scaled = self.global_scaler.transform(s2.reshape(-1, 1)).flatten()
        path = dtw.warping_path(s1_scaled, s2_scaled)
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        
        # But we PLOT the original (unscaled) values for interpretability
        dtwvis.plot_warping(s1, s2, path, fig=fig, axs=(ax1, ax2))
        ax1.set_title("DTW Warping Path Explained", fontsize=16)
        ax1.set_ylabel("Input Patient (Interpolated)")
        ax2.set_ylabel("Best Match Trajectory")
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return image_base64