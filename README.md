# IoT-Enabled Intrusion Detection in Cloud Using Machine Learning

## Scope
Local academic proof-of-concept using a synthetic IoT/cloud network-traffic dataset, Support Vector Machine (RBF kernel) and Streamlit. It does not connect to AWS/Azure/GCP, capture real packets, or block attacks. Synthetic-data metrics are not evidence of real-world security performance.

## Run on Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open http://localhost:8501.

## Dataset
`data/iot_network_traffic.csv` contains 3,000 simulated records. Features: duration_sec, src_bytes, dst_bytes, packet_count, failed_logins, unique_dst_ports, connection_rate, hour, protocol_code. Label: 0=Normal, 1=Intrusion.

## Three review phases
### Phase 1 – Design and dataset
Present the IoT → cloud traffic → ML → alert architecture, dataset preview, features, class counts and data preparation. Output: labelled CSV and dataset analysis.
### Phase 2 – Machine learning
Run `python train_model.py`. Present Support Vector Machine (RBF kernel) training, 75/25 split, accuracy, precision, recall, F1 and confusion matrix. Quote actual measured output.
### Phase 3 – Dashboard integration
Run `streamlit run app.py`. Demonstrate Overview, Live Prediction, Batch Analysis and Model Evaluation. Upload the included CSV and download predictions.

## Final result
The prototype demonstrates an end-to-end simulated IoT/cloud intrusion-detection workflow using Support Vector Machine (RBF kernel) and an interactive Streamlit dashboard. Future work: connect approved real cloud logs, use real IoT traffic datasets, add authentication, secure ingestion and alerting, and validate with device/time-separated test data.
