import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import io
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

app = dash.Dash(__name__)
app.title = "Milestone 4"


app.layout = html.Div([
    html.Div([
        html.Label("Upload File"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        )
    ]),
    html.Div(id='upload-status', style={'color': 'green', 'margin': '10px'}),

    html.Div([
        html.Label("Select Target:"),
        dcc.Dropdown(
            id='select-target',
            placeholder="Select the target variable"
        )
    ], style={'margin': '10px'}),

    html.Div([
        dcc.RadioItems(id='select-categorical', style={'margin': '10px', 'display': 'flex', 'flexDirection': 'row'}),

        html.Div([
            dcc.Graph(id='barchart-average'),
            dcc.Graph(id='barchart-correlation')
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px'})
    ], style={'margin': '10px'}),

    dcc.Store(id='stored-data'),

    html.Div([
        html.Label("Select Features:"),
        dcc.Checklist(id='feature-checkboxes', style={'margin': '10px'}),
        html.Button("Train Model", id='train-button', n_clicks=0, style={'margin': '10px'}),
        dcc.Loading(
            id='train-loading',
            type='default',
            children=html.Div(id='train-output', style={'color': 'blue', 'margin': '10px'})
        )
    ], style={'margin': '20px'}),

    html.Div([
        html.Label("Enter Feature Values for Prediction (In the Feature Checklist order, Seperate with commas):"),
        dcc.Input(id='prediction-input-textbox', type='text', placeholder="example with 2 var: 10,10"),
        html.Button("Predict", id='predict-button', n_clicks=0, style={'margin': '10px'}),
        html.Div(id='prediction-output', style={'color': 'blue', 'margin': '10px'})
    ], style={'margin': '20px'})
])

def convert_words_to_numbers(df):
    labels = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
              'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].str.lower().isin(labels.keys()).any():
            df[col] = df[col].str.lower().map(labels).fillna(df[col])
    return df

@app.callback(
    [Output('stored-data', 'data'), Output('upload-status', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_file_upload(contents, filename):
    if contents is None:
        return dash.no_update, ""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = convert_words_to_numbers(df)  
        return df.to_dict('records'), f"Successfully uploaded: {filename}"
    except Exception as e:
        return dash.no_update, f"Error processing file: {str(e)}"

@app.callback(
    Output('select-target', 'options'),
    Input('stored-data', 'data')
)
def update_dropdown(data):
    if data is None:
        return []

    df = pd.DataFrame(data)

    numeric_columns = df.select_dtypes(include=['number']).columns
    return [{'label': col, 'value': col} for col in numeric_columns]

@app.callback(
    [Output('select-categorical', 'options'),
     Output('barchart-average', 'figure'),
     Output('barchart-correlation', 'figure')],
    [Input('select-target', 'value'),
     Input('select-categorical', 'value')],
    State('stored-data', 'data')
)
def update_barcharts(target, categorical, data):
    if data is None or target is None:
        return [], {}, {}

    df = pd.DataFrame(data)

    df['New Regulations Impacting Aerodynamics'] = df['New Regulations Impacting Aerodynamics'].astype('category')
    df['DRS'] = df['DRS'].astype('category')
    df['Season'] = df['Season'].astype('category')

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    fig_avg = {}
    fig_corr = {}

    if categorical and categorical in categorical_columns:
        avg_values = df.groupby(categorical)[target].mean().reset_index()
        fig_avg = px.bar(avg_values, x=categorical, y=target, title=f"Average {target} by {categorical}", text_auto=True)
        fig_avg.update_layout(yaxis_title=f"{target} (average)")

    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns = [col for col in numeric_columns if col != target]  
    correlations = df[numeric_columns].corrwith(df[target]).abs().sort_values(ascending=False).reset_index()
    correlations = correlations.rename(columns={0: 'Correlation Strength (Absolute Value)', 'index': 'Numerical Variables'})
    fig_corr = px.bar(correlations, x='Numerical Variables', y='Correlation Strength (Absolute Value)', title=f"Correlation Strength of Numerical Variables with {target}", text_auto=True)

    return [{'label': col, 'value': col} for col in categorical_columns], fig_avg, fig_corr

@app.callback(
    Output('feature-checkboxes', 'options'),
    Input('stored-data', 'data')
)
def update_feature_checkboxes(data):
    if data is None:
        return []

    df = pd.DataFrame(data)

    numerical_features = df.select_dtypes(include=['number']).columns

    return [{'label': col, 'value': col} for col in numerical_features]


@app.callback(
    Output('train-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('feature-checkboxes', 'value'),
    State('select-target', 'value')
)
def train_model(n_clicks, data, selected_features, target):
    global trained_model, selected_features_final  

    if n_clicks == 0:
        return "Click 'Train Model' to start."

    if data is None:
        return "Please upload a dataset first."

    if not selected_features:
        return "Please select at least one feature."

    if target is None:
        return "Please select a target variable."

    try:
        df = pd.DataFrame(data)

        if target not in df.columns:
            return f"The selected target variable '{target}' is not in the dataset."

        if any(feature not in df.columns for feature in selected_features):
            return "One or more selected features are not in the dataset."

        X = df[selected_features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=['number']).columns

        preprocessor = ColumnTransformer(
            transformers=[ 
                ('num', SimpleImputer(strategy='mean'), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('regressor', RandomForestRegressor(random_state=15))
        ])

        param_grid = {
            'feature_selection__k': [min(2, len(selected_features)), len(selected_features)],
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        trained_model = grid_search.best_estimator_  
        y_pred = trained_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        selected_features_final = np.array(selected_features)[
            trained_model.named_steps['feature_selection'].get_support()]

        return [
            f"Model trained successfully! R² Score: {r2:.5f}",
            html.Br()]
    except Exception as e:
        return f"An error occurred during training: {str(e)}"

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('feature-checkboxes', 'value'),
    State('select-target', 'value'),
    State('prediction-input-textbox', 'value')
)
def predict_value(n_clicks, selected_features, target, input_values):
    if n_clicks == 0:
        return "Enter feature values and click 'Predict'."

    if not selected_features or trained_model is None:
        return "Please train a model first."

    try:
        if not input_values:
            return "Please enter the feature values."

        input_values = [float(val) if val else np.nan for val in input_values.split(',')]

        if len(input_values) != len(selected_features):
            return f"Please enter {len(selected_features)} values for prediction."

        input_df = pd.DataFrame([input_values], columns=selected_features)

        prediction = trained_model.predict(input_df)[0]

        return f"The predicted value of {target} is: {prediction:.2f}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
