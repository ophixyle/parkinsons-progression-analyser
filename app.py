from flask import Flask, request, render_template
import numpy as np
import plotly
import plotly.graph_objects as go
import json

# --- IMPORT the core logic from your analysis engine ---
from analysis_engine import ParkinsonsAnalyzer, CONFIG

app = Flask(__name__, template_folder='templates')

# --- Instantiate Analyzer once on startup ---
# This is efficient as data loading and model training happens only once.
analyzer = ParkinsonsAnalyzer(CONFIG)

@app.route('/')
def home():
    return render_template('index.html')

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

    # --- Run the full analysis using the ParkinsonsAnalyzer instance ---
    results = analyzer.predict(
        new_patient_series=patient_data, 
        patient_age=patient_age, 
        patient_sex=patient_sex
    )
    
    # Handle cases where the analysis returns an error
    if "error" in results:
        return render_template('index.html', error=results['error'])
    
    # --- Generate Visualizations ---
    forecastGraphJSON = None
    xai_image = None
    
    if results:
        # Create the Forecast Plot
        fig = go.Figure()
        
        input_len = len(results['input_data'])
        forecast_len = len(results['forecast'])
        
        # 1. Plot the User's Input Data
        x_input = np.arange(input_len)
        fig.add_trace(go.Scatter(x=x_input, y=results['input_data'], mode='lines+markers', name='Your Patient\'s Data', line=dict(color='red', width=4)))
        
        # 2. Plot the Synthesized Forecast
        # We stitch the forecast to the last point of the input data for a continuous line
        x_forecast = np.arange(input_len - 1, input_len + forecast_len)
        stitched_forecast = np.concatenate(([results['input_data'][-1]], results['forecast']))
        fig.add_trace(go.Scatter(x=x_forecast, y=stitched_forecast, mode='lines', name='Synthesized Forecast', line=dict(color='green', dash='dash', width=4)))
        
        # 3. Plot the Best Matching Trajectory for context
        best_match_full_traj = results['best_match']['full_trajectory']
        x_full = np.arange(len(best_match_full_traj))
        fig.add_trace(go.Scatter(x=x_full, y=best_match_full_traj, mode='lines', name=f'Best Match (Patient #{results["best_match"]["match_id"]})', line=dict(color='gray', dash='dot')))
        
        # Update layout
        title_text = f"Forecast aligns with '{results['predicted_cohort']}' Cohort"
        fig.update_layout(
            title=title_text,
            xaxis_title="Normalized Time Horizon",
            yaxis_title="Total UPDRS Score",
            legend=dict(x=0.01, y=0.99)
        )
        forecastGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate XAI (DTW alignment) image
        try:
            xai_image = analyzer.generate_dtw_plot(
                results['input_data'],
                results['best_match']['full_trajectory']
            )
        except Exception as e:
            print(f"XAI generation failed: {e}")
            xai_image = None

    return render_template(
        'index.html',
        results=results,
        forecastGraphJSON=forecastGraphJSON,
        xai_image=xai_image
    )

if __name__ == '__main__':
    app.run(debug=True)