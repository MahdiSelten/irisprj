from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

# ========== MODEL SELECTION PAGE ==========
@app.get("/", response_class=HTMLResponse)
async def model_selector():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Select Algorithm</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f0f0f0; }
            h1 { color: #333; }
            select, button {
                padding: 10px 20px;
                font-size: 16px;
                margin: 10px;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #45a049;
            }
            .container {
                background: white;
                display: inline-block;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
        </style>
        <script>
            function goToNextPage() {
                const modelId = document.getElementById("model").value;
                window.location.href = "/model?model_id=" + modelId;
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Select a Machine Learning Algorithm</h1>
            <select id="model">
                <option value="0">Decision Tree</option>
                <option value="1">KNN</option>
                <option value="2">Random Forest</option>
                <option value="3">SVM</option>
            </select>
            <br>
            <button onclick="goToNextPage()">Continue</button>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ========== FORM PAGE ==========
@app.get("/model", response_class=HTMLResponse)
async def model_form(request: Request, model_id: int):
    model_names = ["Decision Tree", "KNN", "Random Forest", "SVM"]
    model_name = model_names[model_id] if 0 <= model_id < len(model_names) else "Unknown"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{model_name} Input</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f9f9f9; }}
            input[type="number"] {{
                padding: 10px;
                font-size: 16px;
                width: 200px;
                margin: 10px;
                border-radius: 6px;
                border: 1px solid #ccc;
            }}
            button {{
                padding: 10px 20px;
                font-size: 16px;
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                margin-top: 20px;
            }}
            button:hover {{ background-color: #0056b3; }}
            .form-container {{
                background: white;
                padding: 40px;
                display: inline-block;
                border-radius: 12px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="form-container">
            <h1>{model_name} Input Form</h1>
            <form method="post" action="/submit">
                <input type="hidden" name="model_id" value="{model_id}">
                <input type="number" step="any" name="sepal_length" placeholder="Sepal Length" required><br>
                <input type="number" step="any" name="sepal_width" placeholder="Sepal Width" required><br>
                <input type="number" step="any" name="petal_length" placeholder="Petal Length" required><br>
                <input type="number" step="any" name="petal_width" placeholder="Petal Width" required><br>
                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ========== MODEL FILES MAP ==========
MODEL_FILES = {
    0: "app/DecisionTreeModel",
    1: "app/KNNModel",
    2: "app/RandomForestModel",
    3: "app/SVCModel",
}

# ========== PREDICTION PAGE ==========
@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    model_id: int = Form(...),
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    model_names = ["Decision Tree", "KNN", "Random Forest", "SVM"]

    if model_id not in MODEL_FILES:
        return HTMLResponse(f"<h1>Invalid model ID: {model_id}</h1>", status_code=400)

    try:
        model = joblib.load(MODEL_FILES[model_id])
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading model: {e}</h1>", status_code=500)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        return HTMLResponse(f"<h1>Error during prediction: {e}</h1>", status_code=500)

    # Optional: Convert prediction to class name
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    class_label = class_names[prediction] if prediction in [0, 1, 2] else prediction

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #eef2f3; }}
            .result-container {{
                background: white;
                display: inline-block;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{ color: #333; }}
            ul {{ text-align: left; display: inline-block; }}
            a {{
                display: inline-block;
                margin-top: 30px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 6px;
            }}
            a:hover {{
                background-color: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="result-container">
            <h1>Prediction Result</h1>
            <p><strong>Algorithm:</strong> {model_names[model_id]}</p>
            <p><strong>Input:</strong></p>
            <ul>
                <li>Sepal Length: {sepal_length}</li>
                <li>Sepal Width: {sepal_width}</li>
                <li>Petal Length: {petal_length}</li>
                <li>Petal Width: {petal_width}</li>
            </ul>
            <p><strong>Prediction:</strong> {class_label}</p>
            <a href="/">Back to algorithm selection</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
