#!/usr/bin/env python3
"""
CNC Production Feeder - 18 FEATURES
Dataset: 7 Days Dataset.xlsx
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("=" * 80)
print("CNC PRODUCTION FEEDER - 18 FEATURES")
print("Dataset: 7 Days Dataset")
print("=" * 80)

# ── Config ────────────────────────────────────────────────────────────────────
WEBSOCKET_PORT = 8765
DATA_FILE      = '7 Days Dataset.xlsx'
MODEL_FILE     = 'MOFGB.pkl'
SEND_INTERVAL  = 2.0

# ── Load Model ────────────────────────────────────────────────────────────────
print("\n[1/3] Loading model...")
try:
    with open(MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
    model  = model_data['model']
    scaler = model_data['scaler']
    print(f"  ✓ Model loaded")
    print(f"  ✓ Model expects: {scaler.n_features_in_} features")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ── Load & Prepare Data ───────────────────────────────────────────────────────
print("\n[2/3] Loading Excel dataset...")
try:
    df_raw = pd.read_excel(DATA_FILE)
    print(f"  ✓ Loaded {len(df_raw):,} rows  |  {len(df_raw.columns)} columns")

    # Sort by timestamp so rolling window is chronologically correct
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

    # Compute speed_roll5 — rolling mean of speed over 5 consecutive samples
    df_raw['speed_roll5'] = df_raw['speed'].rolling(window=5, min_periods=1).mean()
    print(f"  ✓ speed_roll5 computed from speed column")

    # Shuffle for production run
    df = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
    total_anom = df['is_anomaly'].sum()
    print(f"  ✓ Shuffled {len(df):,} samples  "
          f"(Anomalies: {total_anom} = {total_anom/len(df)*100:.1f}%)")

except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ── Process Sample ────────────────────────────────────────────────────────────
def process_sample(row):
    try:
        speed        = float(row['speed'])
        Yload        = float(row['Yload'])
        Xload        = float(row['Xload'])
        Zload        = float(row['Zload'])
        Zact         = float(row['Zact'])
        Xact         = float(row['Xact'])
        Yact         = float(row['Yact'])
        hour         = float(row['hour'])
        day_of_week  = float(row['day_of_week'])
        shift        = float(row['shift'])
        Yload_roll5  = float(row['Yload_roll5'])
        Xload_roll5  = float(row['Xload_roll5'])
        Zload_roll5  = float(row['Zload_roll5'])
        Zact_roll5   = float(row['Zact_roll5'])
        Xact_roll5   = float(row['Xact_roll5'])
        speed_roll5  = float(row['speed_roll5'])
        status_active = 1 if str(row['status']).strip().lower() == 'active' else 0
        tool_id      = int(row['tool_id'])

        features = np.array([[
            speed, Yload, Xload, Zload, Zact, Xact, Yact,
            hour, day_of_week, shift,
            Yload_roll5, Xload_roll5, Zload_roll5, Zact_roll5, Xact_roll5,
            speed_roll5, status_active, tool_id,
        ]])

        X_scaled    = scaler.transform(features)
        prediction  = int(model.predict(X_scaled)[0])
        probability = model.predict_proba(X_scaled)[0]

        is_anomaly    = bool(prediction == 1)
        anomaly_score = float(probability[1] * 100)

        if anomaly_score >= 95:   alert_level = "CRITICAL"
        elif anomaly_score >= 75: alert_level = "WARNING"
        elif anomaly_score >= 50: alert_level = "MONITOR"
        else:                     alert_level = "NORMAL"

        # Use dataset anomaly_type column — values from actual data:
        # servo_overload, tool_wear_trend, spindle_speed_fault,
        # vibration_fault, axis_deviation, none
        raw_atype = str(row.get('anomaly_type', 'none')).strip().lower()
        anomaly_type = None
        if is_anomaly:
            if raw_atype not in ('none', 'nan', ''):
                anomaly_type = raw_atype
            else:
                # Fallback: sensor-derived classification
                if abs(Zload) > 2.5 or abs(Xload) > 2.0 or abs(Yload) > 2.0:
                    anomaly_type = 'servo_overload'
                elif speed < 2200 or speed > 2800:
                    anomaly_type = 'spindle_speed_fault'
                else:
                    anomaly_type = 'axis_deviation'

        # Flagged features list (sigma thresholds)
        flagged = []
        for fname, val in [('Yload', Yload), ('Xload', Xload), ('Zload', Zload),
                           ('Zact', Zact),   ('Xact',  Xact),  ('Yact',  Yact)]:
            if abs(val) > 3.0:
                flagged.append({'feature': fname, 'value': round(val, 2), 'severity': 'high'})
            elif abs(val) > 2.0:
                flagged.append({'feature': fname, 'value': round(val, 2), 'severity': 'medium'})
        if speed < 2200 or speed > 2800:
            flagged.append({'feature': 'speed', 'value': round(speed, 1), 'severity': 'high'})

        return {
            'ts':               datetime.now().isoformat(),
            'anomaly':          is_anomaly,
            'score':            round(anomaly_score, 1),
            'alert_level':      alert_level,
            'anomaly_type':     anomaly_type,
            'flagged_features': flagged,
            'features': {
                'Yload': round(Yload, 3),
                'Xload': round(Xload, 3),
                'Zload': round(Zload, 3),
                'Zact':  round(Zact,  3),
                'Xact':  round(Xact,  3),
                'Yact':  round(Yact,  3),
                'speed': round(speed,  1),
            },
            'tool_id':       tool_id,
            'status_active': status_active,
            'ground_truth':  bool(int(row['is_anomaly'])),
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

# ── WebSocket Server ──────────────────────────────────────────────────────────
clients       = set()
index         = 0
machine_paused = False   # True while waiting for operator FAULT_CLEARED
stats   = {'total': 0, 'anomalies': 0, 'correct': 0,
           'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

async def send_loop():
    global index, stats, machine_paused
    while True:
        # ── HALTED: skip sending, print waiting message every interval ───────
        if machine_paused:
            print(f"  ⛔ HALTED — waiting for operator FAULT CLEARED signal...")
            await asyncio.sleep(SEND_INTERVAL)
            continue

        if clients:
            try:
                row = df.iloc[index]
                msg = process_sample(row)
                if msg:
                    stats['total'] += 1
                    if msg['anomaly']: stats['anomalies'] += 1

                    pred   = msg['anomaly']
                    actual = msg['ground_truth']
                    if pred == actual: stats['correct'] += 1
                    if pred and actual:       stats['tp'] += 1
                    elif pred and not actual: stats['fp'] += 1
                    elif not pred and actual: stats['fn'] += 1
                    else:                     stats['tn'] += 1

                    msg['total']        = stats['total']
                    msg['anomalies']    = stats['anomalies']
                    msg['anomaly_rate'] = round(stats['anomalies'] / stats['total'] * 100, 2)
                    msg['accuracy']     = round(stats['correct']   / stats['total'] * 100, 2)

                    payload = json.dumps(msg)
                    dead = set()
                    for c in clients:
                        try:    await c.send(payload)
                        except: dead.add(c)
                    clients.difference_update(dead)

                    flag = "🚨 ANOMALY" if msg['anomaly'] else "✓ Normal  "
                    mark = "✓" if pred == actual else "✗"
                    acc  = stats['correct'] / stats['total'] * 100
                    print(f"[{stats['total']:5d}] {flag} | Score:{msg['score']:5.1f}% | "
                          f"{msg['alert_level']:8s} | GT:{mark} | Acc:{acc:.1f}%")

                    # ── Auto-pause terminal when anomaly sent ─────────────────
                    if msg['anomaly']:
                        machine_paused = True
                        print(f"\n{'─'*60}")
                        print(f"  ⛔ ANOMALY SENT — MACHINE HALTED")
                        print(f"  Score : {msg['score']}%  Level: {msg['alert_level']}")
                        print(f"  Type  : {msg['anomaly_type']}")
                        print(f"  ➜  Click 'FAULT CLEARED — RESUME' in dashboard")
                        print(f"{'─'*60}\n")

                index = (index + 1) % len(df)
            except Exception as e:
                print(f"  ERROR: {e}")
        await asyncio.sleep(SEND_INTERVAL)

async def handle(ws):
    global machine_paused
    clients.add(ws)
    print(f"\n  ✓ Client connected: {ws.remote_address[0]}")
    try:
        async for msg in ws:
            # Dashboard sends plain text control messages
            if msg == "FAULT_CLEARED":
                machine_paused = False
                print(f"\n{'─'*60}")
                print(f"  ✅ FAULT CLEARED by operator — resuming data feed")
                print(f"{'─'*60}\n")
            elif msg == "FAULT_DETECTED":
                machine_paused = True
                print(f"\n  ⛔ Dashboard confirmed fault detected — staying halted")
    except Exception:
        pass
    finally:
        clients.discard(ws)
        print(f"\n  ✗ Client disconnected")

async def main():
    print("\n[3/3] Starting server...")
    asyncio.create_task(send_loop())
    async with websockets.serve(
        handle,
        "0.0.0.0",
        WEBSOCKET_PORT,
        ping_interval=20,
        ping_timeout=30,
        origins=None,
        max_size=2**20,
    ):
        print(f"\n{'='*80}")
        print(f"  ✓ PRODUCTION SERVER READY!")
        print(f"{'='*80}")
        print(f"  WebSocket : ws://localhost:{WEBSOCKET_PORT}")
        print(f"  Dataset   : {DATA_FILE}  ({len(df):,} samples)")
        print(f"  Interval  : {SEND_INTERVAL}s | Features: 18 (incl. speed_roll5, status_active, tool_id)")
        print(f"  Anomaly types: servo_overload, tool_wear_trend, spindle_speed_fault,")
        print(f"                 vibration_fault, axis_deviation")
        print(f"\n  1. Open dashboard HTML")
        print(f"  2. Enter: localhost:{WEBSOCKET_PORT}")
        print(f"  3. Click CONNECT")
        print(f"\n  Ctrl+C to stop")
        print(f"{'='*80}\n")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("STOPPED")
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  Total:{stats['total']:,}  Anomalies:{stats['anomalies']}  Acc:{acc:.1f}%")
            print(f"  TP:{stats['tp']}  FP:{stats['fp']}  TN:{stats['tn']}  FN:{stats['fn']}")
        print(f"{'='*80}")
