@echo off
echo Starting Heart Disease Prediction Project...
echo ===========================================

echo.
echo Installing dependencies (this may take a minute if not already installed)...
python -m pip install --user -r requirements.txt

echo.
echo Downloading and preparing the dataset...
python data.py

echo.
echo Training the machine learning models...
python model.py

echo.
echo Launching the Streamlit Web Application...
echo The application will open in your default browser automatically.
streamlit run app.py

pause
