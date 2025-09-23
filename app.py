from flask import Flask, request, render_template
import numpy as np
import plotly
import plotly.graph_objects as go
import json

from analysis_engine import ParkinsonsAnalyzer, CONFIG

app = Flask(__name__, template_folder='templates')

analyzer = ParkinsonsAnalyzer(CONFIG)

@app.route('/')
def home():
    # FIX: Pass the request object to the template on the home page.
    return render_template('index.html', request=request)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        patient_data_str = request.form.getlist('updrs')
        patient_data = [float(i) for i in patient_data_str if i]
        
        patient_age_str = request.form.get('age')
        patient_age = int(patient_age_str) if patient_age_str else None

        patient_sex_str = request.form.get('sex')
        patient_sex = int(patient_sex_str) if patient_sex_str else None

        if len(patient_data) < 2:
            return render_template('index.html', error="Please enter at least 2 data points.", request=request)

    except (ValueError, TypeError):
        return render_template('index.html', error="Invalid input. Please enter valid numbers.", request=request)

    results = analyzer.predict(
        new_patient_series=patient_data, 
        patient_age=patient_age, 
        patient_sex=patient_sex
    )
    
    if "error" in results:
        return render_template('index.html', error=results['error'], request=request)
    
    forecastGraphJSON = None
    xai_image = None
    
    if results:
        fig = go.Figure()
        
        input_len = len(results['input_data'])
        forecast_len = len(results['forecast'])
        
        # 1. Plot User's Input Data (with interactive hover text)
        x_input = np.arange(input_len)
        hover_text_input = [f"Visit {i+1}<br>Your UPDRS: {y:.2f}" for i, y in enumerate(results['input_data'])]
        fig.add_trace(go.Scatter(
            x=x_input, y=results['input_data'], mode='lines+markers', name='Your Patient\'s Data', 
            line=dict(color='red', width=4),
            hoverinfo='text', text=hover_text_input
        ))
        
        # 2. Plot Synthesized Forecast (with interactive hover text)
        x_forecast = np.arange(input_len - 1, input_len + forecast_len)
        stitched_forecast = np.concatenate(([results['input_data'][-1]], results['forecast']))
        hover_text_forecast = [f"Forecast Step {i+1}<br>Predicted UPDRS: {y:.2f}" for i, y in enumerate(results['forecast'])]
        hover_text_forecast.insert(0, f"Visit {input_len}<br>Your UPDRS: {results['input_data'][-1]:.2f}")
        fig.add_trace(go.Scatter(
            x=x_forecast, y=stitched_forecast, mode='lines', name='Synthesized Forecast', 
            line=dict(color='green', dash='dash', width=4),
            hoverinfo='text', text=hover_text_forecast
        ))
        
        # 3. Plot Best Matching Trajectory (with interactive hover text)
        best_match_full_traj = results['best_match']['full_trajectory']
        x_full = np.linspace(0, len(x_forecast) -1, len(best_match_full_traj))
        hover_text_match = [f"Time Step {i+1}<br>Match UPDRS: {y:.2f}" for i, y in enumerate(best_match_full_traj)]
        fig.add_trace(go.Scatter(
            x=x_full, y=best_match_full_traj, mode='lines', name=f'Best Match (Patient #{results["best_match"]["match_id"]})', 
            line=dict(color='gray', dash='dot'),
            hoverinfo='text', text=hover_text_match
        ))
        
        title_text = f"Forecast aligns with '{results['predicted_cohort']}' Cohort"
        fig.update_layout(
            title=title_text,
            xaxis_title="Time Steps (Visits)",
            yaxis_title="Total UPDRS Score",
            legend=dict(x=0.01, y=0.99)
        )
        forecastGraphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
        xai_image=xai_image,
        request=request
    )

if __name__ == '__main__':
    app.run(debug=True)