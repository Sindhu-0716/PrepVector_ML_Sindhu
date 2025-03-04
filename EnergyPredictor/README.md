## Folder Structure
- The Data folder has: Train.csv, Test.csv, PJME_hourly.csv (raw data), final_data.csv (data after cleaning and transformation before feeding to model),
- The forecast_test.csv is the file that is used for my static version of the streamlit app - it has limited rows that used to build a mini version of my app
### App folder
- The App.py is the application
- forecast.py has
### Models folder
- Has Prophet Model stored in a pickle file
### Notebooks
- has the requirements.txt with necessary packages to be installed for Streamlit App Deployment
