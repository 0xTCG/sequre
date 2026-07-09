import random
from numpy.create import array, zeros
from sequre import local, Sharetensor as Stensor, sequre

def load_csv(filepath: str):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                row = [float(x) for x in line.strip().split(',')]
                data.append(row)
    return array(data)

# FIX: New function to flatten tall bias columns into flat rows
def load_bias_as_row(filepath: str):
    single_row = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                # Take the first number and put it in our single row
                single_row.append(float(line.strip().split(',')[0]))
    # Wrap it in brackets so it becomes [1, 64] instead of [64, 1]
    return array([single_row])

@sequre
def secure_rnn(mpc, x_enc_list, Wx_enc, Wh_enc, b_enc, Wy_enc, by_enc):
    h = Wx_enc.zeros((1, 64)) 
    for t in range(48):
        xt = x_enc_list[t]
        
        input_part = xt @ Wx_enc
        mem_part = h @ Wh_enc
        # Now h (1x64) and b_enc (1x64) match perfectly!
        h = input_part + mem_part + b_enc
        
    logits = h @ Wy_enc + by_enc
    return logits.reveal(mpc)

@local
def run_sepsis_local(mpc):
    print(f"CP{mpc.pid}: Loading Sepsis Data & Weights...")
    try:
        Wx_raw = load_csv("Wx.csv")
        Wh_raw = load_csv("Wh.csv")
        b_raw  = load_bias_as_row("b_rnn.csv") # Used the new shape fix
        Wy_raw = load_csv("Wy.csv")
        by_raw = load_bias_as_row("by.csv")    # Used the new shape fix
    except Exception as e:
        print(f"Error loading files: {e}")
        return
        
    try:
        raw_data = []
        with open("patient_data.csv", 'r') as f:
            for line in f:
                if line.strip():
                    raw_data.append([float(x) for x in line.strip().split(',')])
    except:
        print("Warning: patient_data.csv not found, using random patient.")
        raw_data = [[random.random() for _ in range(6)] for _ in range(48)]

    print(f"CP{mpc.pid}: Encrypting...")
    Wx_enc = Stensor.enc(mpc, Wx_raw)
    Wh_enc = Stensor.enc(mpc, Wh_raw)
    b_enc  = Stensor.enc(mpc, b_raw) 
    Wy_enc = Stensor.enc(mpc, Wy_raw)
    by_enc = Stensor.enc(mpc, by_raw)
    
    x_enc_list = []
    for i in range(48):
        row_2d = array([raw_data[i]])
        x_enc_list.append(Stensor.enc(mpc, row_2d))

    print(f"CP{mpc.pid}: Running Secure RNN...")
    final_score = secure_rnn(mpc, x_enc_list, Wx_enc, Wh_enc, b_enc, Wy_enc, by_enc)

    print(f"CP{mpc.pid}: FINAL SEPSIS SCORE: {final_score}")
    if final_score[0][0] > 0.0:
        print(f"CP{mpc.pid}: PREDICTION: SEPSIS DETECTED")
    else:
        print(f"CP{mpc.pid}: PREDICTION: HEALTHY")

run_sepsis_local()
