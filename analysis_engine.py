import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Constants and Configuration ---
DATA_FILE = 'parkinsons_updrs.data'
INTERPOLATION_POINTS = 100
COHORT_THRESHOLDS = {
    'Slow Progressor': 15,
    'Moderate Progressor': 25,
    'Rapid Progressor': 40
}

# --- Data Loading and Caching ---
_data_cache = None

def load_data():
    """Loads and preprocesses the dataset from the CSV file."""
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    
    # Group data by patient
    grouped = df.groupby('subject#')
    
    # Store trajectories and metadata
    trajectories = {
        'motor_updrs': {id: data['motor_UPDRS'].values for id, data in grouped},
        'total_updrs': {id: data['total_UPDRS'].values for id, data in grouped}
    }
    
    # Extract age and sex for each patient (taking the first value)
    metadata = {
        'age': {id: data['age'].iloc[0] for id, data in grouped},
        'sex': {id: data['sex'].iloc[0] for id, data in grouped}
    }
    
    _data_cache = (trajectories, metadata)
    return _data_cache

# --- Core Analysis Functions ---

def interpolate_trajectory(trajectory, num_points=INTERPOLATION_POINTS):
    """Interpolates a single trajectory to a fixed number of points."""
    if len(trajectory) < 2:
        return np.array([])
    x_original = np.linspace(0, 1, len(trajectory))
    x_new = np.linspace(0, 1, num_points)
    f = interp1d(x_original, trajectory, kind='linear', fill_value="extrapolate")
    return f(x_new)

def classify_cohort(trajectory):
    """Classifies a patient into a progression cohort based on their UPDRS trajectory."""
    if trajectory is None or len(trajectory) == 0:
        return "Unknown"
    max_score = np.max(trajectory)
    for cohort, threshold in sorted(COHORT_THRESHOLDS.items(), key=lambda item: item[1]):
        if max_score <= threshold:
            return f"{cohort} (UPDRS < {threshold})"
    return f"Very Rapid Progressor (UPDRS >= {COHORT_THRESHOLDS['Rapid Progressor']})"

def find_best_match(new_patient_data, patient_age=None, patient_sex=None, updrs_type='total_updrs'):
    """
    Finds the best matching patient from the dataset using DTW and context filters.
    V6.1: Incorporates age and sex as optional filters.
    """
    trajectories, metadata = load_data()
    
    if not new_patient_data:
        return None

    # Interpolate the new patient's data
    new_patient_interp = interpolate_trajectory(np.array(new_patient_data))
    if new_patient_interp.size == 0:
        return None

    best_match_id = -1
    min_distance = float('inf')
    
    candidate_ids = list(trajectories[updrs_type].keys())

    # --- Context-based Filtering ---
    if patient_age is not None:
        candidate_ids = [pid for pid in candidate_ids if abs(metadata['age'][pid] - patient_age) <= 5]
    if patient_sex is not None:
        candidate_ids = [pid for pid in candidate_ids if metadata['sex'][pid] == patient_sex]

    if not candidate_ids:
        # If filters yield no candidates, return a specific message or relax criteria
        return None # Or handle this case more gracefully

    # --- DTW Comparison ---
    for patient_id in candidate_ids:
        existing_trajectory = trajectories[updrs_type][patient_id]
        if len(existing_trajectory) < 2:
            continue
            
        existing_interp = interpolate_trajectory(existing_trajectory)
        distance = dtw.distance(new_patient_interp, existing_interp, use_pruning=True)
        
        if distance < min_distance:
            min_distance = distance
            best_match_id = patient_id

    if best_match_id == -1:
        return None

    # --- Prepare Results ---
    match_trajectory_full = trajectories[updrs_type][best_match_id]
    match_trajectory_interp = interpolate_trajectory(match_trajectory_full)
    
    cohort = classify_cohort(match_trajectory_full)
    
    # Calculate confidence (heuristic)
    # A lower DTW distance means a better match. We need to normalize this.
    # This is a simple heuristic and can be improved.
    max_possible_dist = len(new_patient_interp) * np.std(new_patient_interp) # A rough upper bound
    confidence = max(0, 100 * (1 - (min_distance / (max_possible_dist + 1e-6))))


    return {
        "match_id": best_match_id,
        "distance": min_distance,
        "confidence": confidence,
        "cohort": cohort,
        "match_trajectory_raw": match_trajectory_full.tolist(),
        "match_trajectory_interp": match_trajectory_interp.tolist(),
        "new_patient_interp": new_patient_interp.tolist()
    }

def dtw_visualisation(new_patient_interp, match_trajectory_interp, best_match_id):
    """
    Generates the DTW path visualization for XAI, matching the notebook's style.
    Returns a Matplotlib figure object.
    """
    s1 = np.asarray(new_patient_interp, dtype=np.double)
    s2 = np.asarray(match_trajectory_interp, dtype=np.double)
    
    path = dtw.warping_path(s1, s2)
    
    # Create a figure with two subplots, just like the notebook
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    
    # Use the dtaidistance plotting function, passing the figure and axes
    dtwvis.plot_warping(s1, s2, path, fig=fig, axs=(ax1, ax2))
    
    ax1.set_title("DTW Warping Path Explained", fontsize=16)
    fig.tight_layout()
    
    return fig
