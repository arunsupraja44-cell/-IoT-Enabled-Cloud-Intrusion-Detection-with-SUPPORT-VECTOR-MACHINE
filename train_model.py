from pathlib import Path
import json, joblib, pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
ROOT=Path(__file__).resolve().parent
FEATURES=["duration_sec","src_bytes","dst_bytes","packet_count","failed_logins","unique_dst_ports","connection_rate","hour","protocol_code"]
df=pd.read_csv(ROOT/"data/iot_network_traffic.csv")
X,y=df[FEATURES],df.label.astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
# SVM is distance-based, so features are scaled first; probability=True enables predict_proba for the dashboard
model=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=10,gamma="scale",class_weight="balanced",probability=True,random_state=42))
model.fit(Xtr,ytr); pred=model.predict(Xte)
m={"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),"recall":float(recall_score(yte,pred,zero_division=0)),"f1":float(f1_score(yte,pred,zero_division=0)),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"train_rows":len(ytr),"test_rows":len(yte)}
(ROOT/"models").mkdir(exist_ok=True)
joblib.dump({"model":model,"features":FEATURES},ROOT/"models/intrusion_model.joblib")
(ROOT/"models/metrics.json").write_text(json.dumps(m,indent=2))
print("Model trained and saved.")
print(f"Accuracy: {m['accuracy']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:",m["confusion_matrix"])
