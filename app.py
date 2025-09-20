from flask import Flask, request, render_template
import numpy as np
import plotly
import plotly.graph_objects as go
import json
import io
import base64

# --- IMPORT the core logic from your analysis engine ---
# This is professional practice: keeps the app separate from the science.
from analysis_engine import find_best_match, dtw_visualisation

app = Flask(__name__, template_folder='templates')

# --- Main route for the web page ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Route to handle the analysis request ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get and clean the data from the web form
        patient_data_str = request.form.getlist('updrs')
        patient_data = [float(i) for i in patient_data_str if i]
        
        patient_age_str = request.form.get('age')
        patient_age = int(patient_age_str) if patient_age_str else None

        patient_sex_str = request.form.get('sex')
        patient_sex = int(patient_sex_str) if patient_sex_str else None

        if len(patient_data) < 2:
            return render_template('index.html', error="Please enter at least 2 data points.")

    except (ValueError, TypeError):
        return render_template('index.html', error="Invalid input. Please enter valid numbers.")

    # --- Run the full V6.1 analysis engine! ---
    results = find_best_match(patient_data, patient_age=patient_age, patient_sex=patient_sex)
    
    # If the engine returns results, generate the visualizations
    if results:
        # Create the Forecast Plot ("Crystal Ball")
        fig1 = go.Figure()
        cutoff_point = 20
        matched_trajectory = results['match_trajectory_interp']
        x_full, x_forecast = np.arange(100), np.arange(cutoff_point - 1, 100)
        fig1.add_trace(go.Scatter(x=np.arange(len(results['new_patient_interp'])), y=results['new_patient_interp'], mode='lines+markers', name='Your Patient\'s Trajectory', line=dict(color='red', width=4)))
        fig1.add_trace(go.Scatter(x=x_full, y=matched_trajectory, mode='lines', name=f'Best Match (Patient #{results["match_id"]})', line=dict(color='gray', dash='dot')))
        fig1.add_trace(go.Scatter(x=x_forecast, y=matched_trajectory[cutoff_point-1:], mode='lines', name='Projected Forecast', line=dict(color='green', dash='dash', width=4)))
        title_text = f"Forecast: Aligns with '{results['cohort'].split('(')[0].strip()}'"
        fig1.update_layout(title=title_text, xaxis_title="Normalized Time", yaxis_title="Interpolated UPDRS Score", legend=dict(x=0.01, y=0.99))
        forecastGraphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate XAI (DTW alignment) image and embed as base64
        try:
            fig_xai = dtw_visualisation(
                results['new_patient_interp'],
                matched_trajectory,
                results['match_id']
            )
            buf = io.BytesIO()
            fig_xai.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            xai_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"XAI generation failed: {e}")
            xai_image = None

        return render_template(
            'index.html',
            results=results,
            forecastGraphJSON=forecastGraphJSON,
            xai_image=xai_image
        )

    else:
        return render_template('index.html', error="Could not find a suitable match based on the provided data and context.")

if __name__ == '__main__':
    app.run(debug=True)