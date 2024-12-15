import dash
import os
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import ALL
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import base64
import io

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose server for Gunicorn
app.title = "Polynomial Ridge Regression App"

# Global variables
uploaded_data = None
model_pipeline = None

# Layout
app.layout = dbc.Container([
    html.H1("Polynomial Ridge Regression App", className="text-center mb-4"),
    
    # Upload Dataset
    html.Div([
        html.H3("Upload Dataset (CSV)", className="mt-3"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
        ),
        html.Div(id='output-data-upload', className="mt-2"),
    ]),
    
    # Select Target Variable
    html.Div([
        html.H3("Select Target Variable"),
        dcc.Dropdown(id='target-dropdown', placeholder="Select Target Variable"),
    ], className="mt-4"),
    
    # Train Model
    html.Div([
        html.H3("Train Polynomial Ridge Regression Model"),
        html.Div(id='feature-checkboxes', className="mt-3"),
        html.Button('Train Model', id='train-button', n_clicks=0, className="btn btn-primary mt-3"),
        html.Div(id='model-score', className="mt-3"),
    ], className="mt-4"),
    
    # Predict
    html.Div([
        html.H3("Predict Target Variable"),
        dcc.Input(id='predict-input', type='text', placeholder="Enter feature values, separated by commas"),
        html.Button('Predict', id='predict-button', n_clicks=0, className="btn btn-success mt-3"),
        html.Div(id='prediction-result', className="mt-3"),
    ], className="mt-4"),
], fluid=True)

# Callbacks
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('target-dropdown', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_file(contents, filename):
    global uploaded_data
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            uploaded_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            numerical_vars = uploaded_data.select_dtypes(include=['number']).columns.tolist()
            return (
                f"Uploaded {filename} successfully!",
                [{'label': col, 'value': col} for col in numerical_vars]
            )
        except Exception as e:
            return f"Error: {e}", []
    return "No file uploaded yet.", []

@app.callback(
    Output('feature-checkboxes', 'children'),
    Input('target-dropdown', 'value')
)
def update_feature_checkboxes(target_var):
    if target_var:
        feature_options = [col for col in uploaded_data.columns if col != target_var]
        return [
            dbc.Checkbox(
                id={'type': 'feature', 'index': i},  # Properly formatted ID
                label=col,
                value=True,
                style={'margin': '5px'}
            ) for i, col in enumerate(feature_options)
        ]
    return []

@app.callback(
    Output('model-score', 'children'),
    Input('train-button', 'n_clicks'),
    State('target-dropdown', 'value'),
    State({'type': 'feature', 'index': ALL}, 'value')
)
def train_model(n_clicks, target_var, feature_values):
    global model_pipeline
    if n_clicks > 0 and target_var:
        selected_features = [col for col, val in zip(uploaded_data.columns, feature_values) if val]
        if not selected_features:
            return "Error: Please select at least one feature."
        
        # Prepare dataset
        X = uploaded_data[selected_features]
        y = uploaded_data[target_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing pipeline
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        # Polynomial Ridge Regression
        poly = PolynomialFeatures(degree=2, include_bias=False)
        ridge_params = {'alpha': [0.1, 1, 10, 100, 1000]}
        ridge = Ridge()
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('poly', poly),
            ('ridge_cv', GridSearchCV(ridge, ridge_params, cv=5, scoring='r2'))
        ])
        
        # Train model
        model_pipeline.fit(X_train, y_train)

        # Model evaluation
        best_ridge = model_pipeline.named_steps['ridge_cv'].best_estimator_
        X_test_poly = poly.transform(preprocessor.transform(X_test))
        y_pred = best_ridge.predict(X_test_poly)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        try:
            # Calculate RMSE
            rmse = mean_squared_error(y_test, y_pred, squared=False)  # Requires scikit-learn >= 0.22
        except TypeError:
            # Fallback for older versions
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5

        return f"Model trained with R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}"
    return "No model trained yet."

@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value')
)
def predict_value(n_clicks, input_values):
    if n_clicks > 0 and input_values:
        try:
            # Parse the input values into a numpy array
            input_data = np.array([float(val) for val in input_values.split(',')]).reshape(1, -1)
            
            # Ensure input is in a DataFrame format with appropriate column names
            selected_features = [col for col in uploaded_data.columns if col != 'Season' and col != 'Overtakes']
            input_df = pd.DataFrame(input_data, columns=selected_features)
            
            # Preprocess input using the pipeline
            preprocessed_input = model_pipeline.named_steps['preprocessor'].transform(input_df)
            
            # Apply polynomial features
            poly_input = model_pipeline.named_steps['poly'].transform(preprocessed_input)

            # Predict using the trained model
            best_ridge = model_pipeline.named_steps['ridge_cv'].best_estimator_
            prediction = best_ridge.predict(poly_input)

            return f"Predicted Target Value: {prediction[0]:.2f}"
        except Exception as e:
            return f"Error: {e}"
    return "No prediction made yet."

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
