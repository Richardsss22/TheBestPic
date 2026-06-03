# AF Recurrence Risk Predictor

Research PoC Streamlit app for AF recurrence risk inference.

Run from this folder:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run streamlit_app_v16_poc.py
```

Required local files are kept in:

- `app_artifacts/`
- `results_comp_modelos/final_options/`
- `ml_v16_train.py`
- `ml_v16_extract.py`

Disclaimer: research demonstration only. Not validated for standalone clinical decision-making.

Notes:

- The included validation score table is anonymized and does not include patient IDs.
- Do not commit local patient folders, raw `.mat` files, generated reports, Firebase credentials, or `.env` secrets.
