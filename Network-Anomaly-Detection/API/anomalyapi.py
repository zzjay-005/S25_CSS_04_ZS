import os
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List

# Get the project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models and preprocessing tools
rf_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "network_anomaly_model_rf.pkl"))
lr_model = joblib.load(os.path.join(PROJECT_ROOT, "models", "network_anomaly_model_lr.pkl"))
scaler = joblib.load(os.path.join(PROJECT_ROOT, "models", "scaler_lr.pkl"))
imputer = joblib.load(os.path.join(PROJECT_ROOT, "models", "imputer_lr.pkl"))
encoder = joblib.load(os.path.join(PROJECT_ROOT, "models", "encoder_lr.pkl"))

# Manually defined feature list from model training
ALL_FEATURES = [
    "pkSeqID", "proto", "saddr", "sport", "daddr", "dport", "seq", "stddev",
    "N_IN_Conn_P_SrcIP", "min", "state_number", "mean", "N_IN_Conn_P_DstIP",
    "drate", "srate", "max", "type", "subcategory",
    "ÿid", "dur", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
    "ÿsrc_ip", "src_port", "dst_ip", "dst_port", "duration", "src_bytes", "dst_bytes",
    "conn_state", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_query", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA",
    "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "ssl_subject", "ssl_issuer", "http_trans_depth", "http_method", "http_uri",
    "http_version", "http_request_body_len", "http_response_body_len", "http_status_code",
    "http_user_agent", "http_orig_mime_types", "http_resp_mime_types", "weird_name",
    "weird_addl", "weird_notice"
]



# Create FastAPI app
app = FastAPI(title="Network Anomaly Detection API")

# Define input schema
class InputData(BaseModel):
    data: List[dict]

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)

        # Encode object/categorical columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = encoder.transform(df[col].astype(str))

        # Fill missing features with 0s
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training order
        df = df[ALL_FEATURES]

        # Impute and scale
        df_imputed = pd.DataFrame(imputer.transform(df), columns=ALL_FEATURES)
        df_scaled = scaler.transform(df_imputed)

        # Predict
        preds = lr_model.predict(df_scaled)
        probs = lr_model.predict_proba(df_scaled)[:, 1]

        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Network Anomaly Detection API is running "}


