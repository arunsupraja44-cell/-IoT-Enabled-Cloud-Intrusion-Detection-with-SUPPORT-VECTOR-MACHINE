from pathlib import Path
import json, joblib
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
FEATURES=["duration_sec","src_bytes","dst_bytes","packet_count","failed_logins","unique_dst_ports","connection_rate","hour","protocol_code"]
HELP=["Duration (seconds)","Source bytes","Destination bytes","Packet count","Failed logins","Unique destination ports","Connection rate","Hour (0-23)","Protocol code (0=TCP, 1=UDP, 2=MQTT/Other)"]
if not (ROOT/"models/intrusion_model.joblib").exists():
    import train_model
bundle=joblib.load(ROOT/"models/intrusion_model.joblib"); model=bundle["model"]
df=pd.read_csv(ROOT/"data/iot_network_traffic.csv")
metrics=json.loads((ROOT/"models/metrics.json").read_text())
st.set_page_config(page_title="IoT Cloud Intrusion Detection",page_icon="🛡️",layout="wide")
st.title("🛡️ IoT-Enabled Cloud Intrusion Detection")
st.caption("Academic prototype | Simulated IoT traffic | Support Vector Machine")
page=st.sidebar.radio("Navigation",["Overview","Live Prediction","Batch Analysis","Model Evaluation","Project Review"])
st.sidebar.caption("Simulation only: not connected to live cloud infrastructure or packet capture.")
if page=="Overview":
    st.subheader("Security monitoring overview")
    a,b,c,d=st.columns(4)
    a.metric("Traffic records",f"{len(df):,}")
    b.metric("Normal records",f"{(df.label==0).sum():,}")
    c.metric("Intrusion records",f"{(df.label==1).sum():,}")
    d.metric("Attack share",f"{df.label.mean():.1%}")
    fig,ax=plt.subplots(); df.label.map({0:"Normal",1:"Intrusion"}).value_counts().plot(kind="bar",ax=ax); ax.set_ylabel("Records"); st.pyplot(fig); plt.close(fig)
    st.info("Synthetic data is used for academic demonstration. Real-world performance must be validated on representative network data.")
elif page=="Live Prediction":
    st.subheader("Single traffic-record inspection")
    presets={
    "Normal IoT sensor":[5.2,950,1100,16,0,2,3.5,14,0],
    "Suspicious port scan":[1,300,180,120,0,18,85,2,0],
    "Brute-force login":[3,250,90,35,18,1,22,1,0],
    "High-volume DoS-like traffic":[.4,4200,80,240,0,2,180,3,1]}
    choice=st.selectbox("Example",list(presets))
    vals=presets[choice]
    with st.form("prediction"):
        entered={}
        cols=st.columns(3)
        for i,(f,label) in enumerate(zip(FEATURES,HELP)):
            with cols[i%3]:
                if f in ["packet_count","failed_logins","unique_dst_ports","hour","protocol_code"]:
                    entered[f]=st.number_input(label,min_value=0,value=int(vals[i]),step=1)
                else:
                    entered[f]=st.number_input(label,min_value=0.0,value=float(vals[i]),step=1.0)
        go=st.form_submit_button("Analyze traffic",type="primary")
    if go:
        x=pd.DataFrame([[entered[f] for f in FEATURES]],columns=FEATURES)
        pred=int(model.predict(x)[0]); prob=float(model.predict_proba(x)[0][1])
        if pred: st.error(f"INTRUSION SUSPECTED — attack probability {prob:.1%}")
        else: st.success(f"NORMAL TRAFFIC — attack probability {prob:.1%}")
        st.progress(prob); st.dataframe(x)
elif page=="Batch Analysis":
    st.subheader("Batch CSV prediction")
    uploaded=st.file_uploader("Upload CSV with all nine feature columns",type=["csv"])
    if uploaded:
        incoming=pd.read_csv(uploaded); missing=[f for f in FEATURES if f not in incoming.columns]
        if missing: st.error("Missing columns: "+", ".join(missing))
        else:
            x=incoming[FEATURES].apply(pd.to_numeric,errors="coerce")
            if x.isna().any().any(): st.error("Feature values must all be numeric and non-empty.")
            else:
                pred=model.predict(x); out=incoming.copy()
                out["prediction"]=np.where(pred==1,"Intrusion","Normal")
                out["attack_probability"]=model.predict_proba(x)[:,1].round(4)
                st.metric("Intrusions flagged",int((pred==1).sum())); st.dataframe(out)
                st.download_button("Download predictions",out.to_csv(index=False).encode(), "intrusion_predictions.csv","text/csv")
    else: st.caption("Try data/iot_network_traffic.csv.")
elif page=="Model Evaluation":
    st.subheader("Held-out test-set performance")
    a,b,c,d=st.columns(4)
    for col,key,label in zip([a,b,c,d],["accuracy","precision","recall","f1"],["Accuracy","Precision","Recall","F1-score"]): col.metric(label,f"{metrics[key]:.3f}")
    cm=np.array(metrics["confusion_matrix"]); fig,ax=plt.subplots(); ax.imshow(cm)
    ax.set_xticks([0,1],["Predicted normal","Predicted intrusion"]); ax.set_yticks([0,1],["Actual normal","Actual intrusion"])
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    ax.set_title("Confusion matrix"); st.pyplot(fig); plt.close(fig)
    st.caption(f"Training rows: {metrics['train_rows']} | Test rows: {metrics['test_rows']}")
elif page=="Project Review":
    t1,t2,t3=st.tabs(["Phase 1 – Dataset","Phase 2 – ML model","Phase 3 – Prototype"])
    with t1:
        st.write("Prepare labelled simulated IoT/cloud traffic; 0 = Normal, 1 = Intrusion.")
        st.write(f"{len(df)} records, {len(FEATURES)} input features."); st.dataframe(df.head(10))
    with t2:
        st.write("Support Vector Machine classifier trained with a stratified 75/25 train/test split.")
        st.json({k:metrics[k] for k in ["accuracy","precision","recall","f1","confusion_matrix"]})
    with t3:
        st.write("Demonstrate single-record predictions, batch CSV analysis, dashboard metrics and downloadable prediction results.")
        st.success("Streamlit integration module ready.")
