import h5py
import numpy as np
import scipy.signal as sig

def create_fake_carto_mat(filename="fake_patient_123.mat"):
    # Generate a fake ECG signal (Lead II-like)
    fs = 1000
    t = np.arange(0, 5, 1/fs)
    ecg = np.zeros_like(t)
    
    # Add simulated heartbeats (roughly 60 bpm)
    for beat_time in [0.5, 1.5, 2.5, 3.5, 4.5]:
        idx = int(beat_time * fs)
        if idx+500 < len(ecg):
            p_len = 100
            p_wave = sig.windows.hann(p_len) * 0.15
            ecg[idx-150:idx-150+p_len] += p_wave
            
            qrs = [ -0.1, 1.0, -0.2 ]
            ecg[idx:idx+3] += qrs
            
            t_len = 150
            t_wave = sig.windows.hann(t_len) * 0.25
            ecg[idx+100:idx+100+t_len] += t_wave
            
    ecg += np.random.normal(0, 0.01, len(ecg))
    
    ecg_matrix = np.zeros((12, len(t), 1))
    ecg_matrix[1, :, 0] = ecg  # Lead II at index 1
    
    with h5py.File(filename, "w") as f:
        elec = f.create_group("userdata/electric")
        
        # 1. ecg
        elec.create_dataset("ecg", data=ecg_matrix)
        
        # 2. egm (dummy)
        elec.create_dataset("egm", data=np.zeros((1, 1, 1)))
        
        # 3. voltages/bipolar (dummy)
        voltages = elec.create_group("voltages")
        voltages.create_dataset("bipolar", data=np.zeros((1, 1)))
        
        # 4. voltages/unipolar (dummy)
        voltages.create_dataset("unipolar", data=np.zeros((1, 1)))
        
        # 5. annotations/mapAnnot (dummy)
        annotations = elec.create_group("annotations")
        annotations.create_dataset("mapAnnot", data=np.zeros((1, 1)))
        
        # 6. ecgNames
        names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        refs = []
        for i, name in enumerate(names):
            char_array = np.array([ord(c) for c in name], dtype=np.uint16)
            ds = f.create_dataset(f"name_{i}", data=char_array)
            refs.append(ds.ref)
            
        ref_array = np.array(refs, dtype=h5py.ref_dtype).reshape(-1, 1)
        elec.create_dataset("ecgNames", data=ref_array)

if __name__ == "__main__":
    create_fake_carto_mat()
    print("Created fake_patient_123.mat successfully with all required fields!")
