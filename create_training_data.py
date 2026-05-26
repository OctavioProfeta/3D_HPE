import json, os, shutil, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt

def iqr_roll_median(df, window, k):
    df_clean = df.copy()

    #Selecting only columns with continuous numeric values, excluding time
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    num_cols = [col for col in num_cols if col != "time"]

    #Calculating IQR
    Q1 = df_clean[num_cols].quantile(0.25)
    Q3 = df_clean[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    #Replacing outliers by a rolling median
    for col in num_cols:
        lower = Q1[col] - k * IQR[col]
        upper = Q3[col] + k * IQR[col]

        outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
        rolling_med = df_clean[col].rolling(window=window, center=True, min_periods=1).median()

        df_clean.loc[outliers, col] = rolling_med[outliers]
        
    return df_clean


def angle_between_points(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def angle_between_vectors(v1, v2):
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def frontal_plane_normal_vector(left_hip, right_hip, left_shoulder, right_shoulder):
# This computes the normalized frontal plane normal vector using the hip and shoulder landmarks.
# It calculates the mid-point of the hips and shoulders, then creates two vectors from 
# the left hip to the mid-point and from the right hip to the mid-point.
    mid_point = np.array([(left_hip[0] + right_hip[0] + left_shoulder[0] + right_shoulder[0]) / 4, 
                         (left_hip[1] + right_hip[1] + left_shoulder[1] + right_shoulder[1]) / 4, 
                         (left_hip[2] + right_hip[2] + left_shoulder[2] + right_shoulder[2]) / 4])
    v1 = np.array([left_hip[0] - mid_point[0], left_hip[1] - mid_point[1], left_hip[2] - mid_point[2]])
    v2 = np.array([right_hip[0] - mid_point[0], right_hip[1] - mid_point[1], right_hip[2] - mid_point[2]])
    normal_vector = np.cross(v1, v2)
    return normal_vector / np.linalg.norm(normal_vector)


def sagittal_plane_normal_vector(left_hip, right_hip):
# This compute the normalized sagittal plane normal vector using the hip landmarks. 
# It creates a vector from the right hip to the left hip, which is perpendicular to the sagittal plane.
    normal_vector = np.array([left_hip[0] - right_hip[0], left_hip[1] - right_hip[1], left_hip[2] - right_hip[2]])
    return normal_vector / np.linalg.norm(normal_vector)


def projection_onto_plane(v1, v2, n):
# This would normaly need to pass a point on the plane, but since both planes passes through the world coordinate
# origin (mid_hip point), we don't need to pass it as an argument.
    v1_proj = v1 - np.dot(v1, n) * n / np.dot(n, n)
    v2_proj = v2 - np.dot(v2, n) * n / np.dot(n, n)
    return v1_proj, v2_proj


def calculate_angles(df):
    for i in df.index:
        
        left_shoulder = (df.loc[i,'LEFT_SHOULDER_x'], df.loc[i,'LEFT_SHOULDER_y'], df.loc[i,'LEFT_SHOULDER_z'])
        right_shoulder = (df.loc[i,'RIGHT_SHOULDER_x'], df.loc[i,'RIGHT_SHOULDER_y'], df.loc[i,'RIGHT_SHOULDER_z'])
        left_hip = (df.loc[i,'LEFT_HIP_x'], df.loc[i,'LEFT_HIP_y'], df.loc[i,'LEFT_HIP_z'])
        right_hip = (df.loc[i,'RIGHT_HIP_x'], df.loc[i,'RIGHT_HIP_y'], df.loc[i,'RIGHT_HIP_z'])
        left_knee = (df.loc[i,'LEFT_KNEE_x'], df.loc[i,'LEFT_KNEE_y'], df.loc[i,'LEFT_KNEE_z'])
        right_knee = (df.loc[i,'RIGHT_KNEE_x'], df.loc[i,'RIGHT_KNEE_y'], df.loc[i,'RIGHT_KNEE_z'])
        left_ankle = (df.loc[i,'LEFT_ANKLE_x'], df.loc[i,'LEFT_ANKLE_y'], df.loc[i,'LEFT_ANKLE_z'])
        right_ankle = (df.loc[i,'RIGHT_ANKLE_x'], df.loc[i,'RIGHT_ANKLE_y'], df.loc[i,'RIGHT_ANKLE_z'])

        frontal_vector = frontal_plane_normal_vector(left_hip, right_hip, left_shoulder, right_shoulder)
        sagittal_vector = sagittal_plane_normal_vector(left_hip, right_hip)

        right_femur = np.array([right_knee[0] - right_hip[0], right_knee[1] - right_hip[1], right_knee[2] - right_hip[2]])
        left_femur = np.array([left_knee[0] - left_hip[0], left_knee[1] - left_hip[1], left_knee[2] - left_hip[2]])
        right_tibia = np.array([right_ankle[0] - right_knee[0], right_ankle[1] - right_knee[1], right_ankle[2] - right_knee[2]])
        left_tibia = np.array([left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1], left_ankle[2] - left_knee[2]])

        left_femur_proj_frontal, left_tibia_proj_frontal = projection_onto_plane(left_femur, left_tibia, frontal_vector)
        right_femur_proj_frontal, right_tibia_proj_frontal = projection_onto_plane(right_femur, right_tibia, frontal_vector)
        left_femur_proj_sagittal, left_tibia_proj_sagittal = projection_onto_plane(left_femur, left_tibia, sagittal_vector)
        right_femur_proj_sagittal, right_tibia_proj_sagittal = projection_onto_plane(right_femur, right_tibia, sagittal_vector)

        df.loc[i,'LEFT_ABDUCTION_ANGLE'] = angle_between_vectors(left_femur_proj_frontal, left_tibia_proj_frontal)
        df.loc[i,'RIGHT_ABDUCTION_ANGLE'] = angle_between_vectors(right_femur_proj_frontal, right_tibia_proj_frontal)
        df.loc[i,'LEFT_FLEXION_ANGLE'] = angle_between_vectors(left_femur_proj_sagittal, left_tibia_proj_sagittal)
        df.loc[i,'RIGHT_FLEXION_ANGLE'] = angle_between_vectors(right_femur_proj_sagittal, right_tibia_proj_sagittal)

    return df


def butterworth_filter_df(df, cutoff=10, fs=50, order=2):
    """Apply a zero-phase low-pass Butterworth filter to all landmark x/y/z columns."""
    pos_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_z'))]

    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)

    df_filtered = df.copy()
    for col in pos_cols:
        series = df_filtered[col].copy()
        nan_mask = series.isna()

        # Interpolate NaNs so filtfilt can run, then restore NaNs after filtering.
        series_interp = series.interpolate(method='linear').bfill().ffill()

        if len(series_interp.dropna()) > 3 * max(len(a), len(b)):
            df_filtered[col] = filtfilt(b, a, series_interp.values)
        else:
            df_filtered[col] = series_interp.values

        df_filtered.loc[nan_mask, col] = np.nan

    return df_filtered



# Needed folder to run the notebook
sto_folder = 'ACL_Lengths/'
json_folder = 'landmarks_summary/ATH_videos_avi'
video_folder = 'ATH_videos_avi_processed/'

new_json_folder = 'matched_json/'
new_sto_folder = 'matched_acl_length/'
new_video_folder = 'matched_videos/'
new_mot_folder = 'matched_mot/'

mismatch = 0
count = 0
for ath in sorted(os.listdir(json_folder)):
    for session in sorted(os.listdir(os.path.join(json_folder, ath))):
        for json_file in sorted(os.listdir(os.path.join(json_folder, ath, session))):
            count += 1
            json_path = os.path.join(json_folder, ath, session, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            frame_count = data['frame_count']
            name = json_file[:9].replace(' ', '')
            if name[6] == '.':
                name = name[:6]
            name = name.replace('0', '')
            name = name.replace('.', '')
            sto_file = os.path.join(sto_folder, ath, session, name + '_MuscleAnalysis_Length.sto')
            sto = pd.read_csv(sto_file, sep=r"\s+", skiprows=11)
            if abs(len(sto)/2 - frame_count) > 2:
                print(f'{ath}/{session}/{name}: frame_count={frame_count}, sto_frames={len(sto)/2}')
                mismatch += 1
                continue
            os.makedirs(os.path.join(new_json_folder, ath, session), exist_ok=True)
            os.makedirs(os.path.join(new_sto_folder, ath, session), exist_ok=True)
            os.makedirs(os.path.join(new_video_folder, ath, session), exist_ok=True)
            os.makedirs(os.path.join(new_mot_folder, ath, session), exist_ok=True)
            shutil.copy(json_path, os.path.join(new_json_folder, ath, session, name + '.json'))
            shutil.copy(sto_file, os.path.join(new_sto_folder, ath, session, name + '.sto'))
            video_file = os.path.join(video_folder, ath, session, json_file.replace('_results.json', '.mp4'))
            shutil.copy(video_file, os.path.join(new_video_folder, ath, session, name + '.mp4'))
            mot_file = os.path.join(sto_folder, ath, session, name + '.mot')
            shutil.copy(mot_file, os.path.join(new_mot_folder, ath, session, name + '.mot'))
print(f'Total files processed: {count}')
print(f'Total mismatches: {mismatch}')
print(f'Percentage of mismatches: {mismatch/count*100:.2f}%')



json_folder = 'matched_json/'
acl_folder = 'matched_acl_length/'
mot_folder = 'matched_mot/'

output_folder = 'Data Training Clean/'

skipped = {}
DERIVATIVE_THRESHOLD = 0.04

manually_banned_files = ['ATH10/s2/PreCut1', 'ATH10/s2/PreCut5', 'ATH11/s2/PreCut2', 'ATH14/s1/PreCut', 'ATH18/s1/PreCut'] # Example of manually banned files, format should be 'ath/session/trial'

for ath in sorted(os.listdir(json_folder)):
    for session in sorted(os.listdir(os.path.join(json_folder, ath))):
        for trial in sorted(os.listdir(os.path.join(json_folder, ath, session))):
            print(f'Processing {ath} {session} {trial}...')
            name = f'{ath}/{session}/{trial.replace(".json", "")}'
            
            if name in manually_banned_files:
                print(f'Warning: {name} is manually banned.')
                skipped.update({name: 'Manually banned'})
                continue
            
            # Load JSON data
            json_path = os.path.join(json_folder, name + '.json')
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # Find the indices of where the force on the plates are non null
            mot_file = os.path.join(mot_folder, name + '.mot')
            mot = pd.read_csv(mot_file, sep=r"\s+", skiprows=5)
            condition = (mot["ground_force_px"] > 0) | (mot["1_ground_force_px"] > 0)
            true_indices = np.where(condition)[0] / 2
            start_idx = int(min(true_indices) - 150)
            end_idx = int(max(true_indices) + 20)

            json_frames = json_data['frames'][start_idx:end_idx]
            # raise(f'{ath} {session} {trial}: start_idx={start_idx}, end_idx={end_idx}, true_indices={true_indices}')
            
            # Load ACL length data and get the strain list
            sto_file = os.path.join(acl_folder, name + '.sto')
            sto = pd.read_csv(sto_file, sep=r"\s+", skiprows=11)
            sto = iqr_roll_median(sto, window=5, k=1)

            l_zero_idx = start_idx - 20
            if l_zero_idx % 2 != 0:
                l_zero_idx -= 1

            right_acl_length = sto['acl_r']
            right_acl_length = right_acl_length[::2]
            l_zero_right = np.median(right_acl_length[l_zero_idx-10:l_zero_idx+10])
            right_acl_strain = 1 + (right_acl_length - l_zero_right)/l_zero_right
            right_acl_strain = right_acl_strain[start_idx:end_idx].tolist()
            left_acl_length = sto['acl_l']
            left_acl_length = left_acl_length[::2]
            l_zero_left = np.median(left_acl_length[l_zero_idx-10:l_zero_idx+10])
            left_acl_strain = 1 + (left_acl_length - l_zero_left)/l_zero_left
            left_acl_strain = left_acl_strain[start_idx:end_idx].tolist()

            # Check for extreme change in ACL strain values, which would indicate an error in the data
            right_acl_strain_derivate = np.diff(right_acl_strain)
            left_acl_strain_derivate = np.diff(left_acl_strain)
            
            if all(abs(right_acl_strain_derivate) < DERIVATIVE_THRESHOLD) == False or all(abs(left_acl_strain_derivate) < DERIVATIVE_THRESHOLD) == False:
                print(f'Warning: Extreme change in ACL strain values for {ath} {session} {trial}')
                skipped.update({name: 'Extreme change in ACL strain values'})
                continue

            if right_acl_strain == [] or left_acl_strain == []:
                print(f'Warning: No ACL strain data for {ath} {session} {trial}')
                skipped.update({name: 'No ACL strain data'})
                continue
            if max(right_acl_strain) > 1.35 or min(right_acl_strain) < 0.65 or max(left_acl_strain) > 1.35 or min(left_acl_strain) < 0.65:
                print(f'Warning: Unusual ACL strain values for {ath} {session} {trial}')
                skipped.update({name: 'Unusual ACL strain values'})
                continue

            final_df = pd.DataFrame({'LEFT_SHOULDER_x': [], 'LEFT_SHOULDER_y': [], 'LEFT_SHOULDER_z': [],
                                     'RIGHT_SHOULDER_x': [], 'RIGHT_SHOULDER_y': [], 'RIGHT_SHOULDER_z': [],
                                     'LEFT_HIP_x': [], 'LEFT_HIP_y': [], 'LEFT_HIP_z': [],
                                     'RIGHT_HIP_x': [], 'RIGHT_HIP_y': [], 'RIGHT_HIP_z': [],
                                     'LEFT_KNEE_x': [], 'LEFT_KNEE_y': [], 'LEFT_KNEE_z': [],
                                     'RIGHT_KNEE_x': [], 'RIGHT_KNEE_y': [], 'RIGHT_KNEE_z': [],
                                     'LEFT_ANKLE_x': [], 'LEFT_ANKLE_y': [], 'LEFT_ANKLE_z': [],
                                     'RIGHT_ANKLE_x': [], 'RIGHT_ANKLE_y': [], 'RIGHT_ANKLE_z': [],
                                     'LEFT_ABDUCTION_ANGLE': [], 'RIGHT_ABDUCTION_ANGLE': [],
                                     'LEFT_FLEXION_ANGLE': [], 'RIGHT_FLEXION_ANGLE': [],
                                     'RIGHT_ACL_STRAIN': [], 'LEFT_ACL_STRAIN': [],
                                     'START_FRAME': [], 'END_FRAME': []})

            for i in range(len(right_acl_strain)):
                frame_data = json_frames[i]
                if json_frames[i]['landmarks'] != []:
                    row = {landmark['name']: landmark for landmark in frame_data['landmarks']}
                    row = {
                        'LEFT_SHOULDER_x': row['LEFT_SHOULDER']['x'],
                        'LEFT_SHOULDER_y': row['LEFT_SHOULDER']['y'],
                        'LEFT_SHOULDER_z': row['LEFT_SHOULDER']['z'],
                        'RIGHT_SHOULDER_x': row['RIGHT_SHOULDER']['x'],
                        'RIGHT_SHOULDER_y': row['RIGHT_SHOULDER']['y'],
                        'RIGHT_SHOULDER_z': row['RIGHT_SHOULDER']['z'],
                        'LEFT_HIP_x': row['LEFT_HIP']['x'],
                        'LEFT_HIP_y': row['LEFT_HIP']['y'],
                        'LEFT_HIP_z': row['LEFT_HIP']['z'],
                        'RIGHT_HIP_x': row['RIGHT_HIP']['x'],
                        'RIGHT_HIP_y': row['RIGHT_HIP']['y'],
                        'RIGHT_HIP_z': row['RIGHT_HIP']['z'],
                        'LEFT_KNEE_x': row['LEFT_KNEE']['x'],
                        'LEFT_KNEE_y': row['LEFT_KNEE']['y'],
                        'LEFT_KNEE_z': row['LEFT_KNEE']['z'],
                        'RIGHT_KNEE_x': row['RIGHT_KNEE']['x'],
                        'RIGHT_KNEE_y': row['RIGHT_KNEE']['y'],
                        'RIGHT_KNEE_z': row['RIGHT_KNEE']['z'],
                        'LEFT_ANKLE_x': row['LEFT_ANKLE']['x'],
                        'LEFT_ANKLE_y': row['LEFT_ANKLE']['y'],
                        'LEFT_ANKLE_z': row['LEFT_ANKLE']['z'],
                        'RIGHT_ANKLE_x': row['RIGHT_ANKLE']['x'],
                        'RIGHT_ANKLE_y': row['RIGHT_ANKLE']['y'],
                        'RIGHT_ANKLE_z': row['RIGHT_ANKLE']['z'],
                        'RIGHT_ACL_STRAIN': right_acl_strain[i],
                        'LEFT_ACL_STRAIN': left_acl_strain[i],
                        'START_FRAME': start_idx,
                        'END_FRAME': end_idx
                    }
                else:
                    row = {
                        'LEFT_SHOULDER_x': np.nan,
                        'LEFT_SHOULDER_y': np.nan,
                        'LEFT_SHOULDER_z': np.nan,
                        'RIGHT_SHOULDER_x': np.nan,
                        'RIGHT_SHOULDER_y': np.nan,
                        'RIGHT_SHOULDER_z': np.nan,
                        'LEFT_HIP_x': np.nan,
                        'LEFT_HIP_y': np.nan,
                        'LEFT_HIP_z': np.nan,
                        'RIGHT_HIP_x': np.nan,
                        'RIGHT_HIP_y': np.nan,
                        'RIGHT_HIP_z': np.nan,
                        'LEFT_KNEE_x': np.nan,
                        'LEFT_KNEE_y': np.nan,
                        'LEFT_KNEE_z': np.nan,
                        'RIGHT_KNEE_x': np.nan,
                        'RIGHT_KNEE_y': np.nan,
                        'RIGHT_KNEE_z': np.nan,
                        'LEFT_ANKLE_x': np.nan,
                        'LEFT_ANKLE_y': np.nan,
                        'LEFT_ANKLE_z': np.nan,
                        'RIGHT_ANKLE_x': np.nan,
                        'RIGHT_ANKLE_y': np.nan,
                        'RIGHT_ANKLE_z': np.nan,
                        'RIGHT_ACL_STRAIN': np.nan,
                        'LEFT_ACL_STRAIN': np.nan,
                        'START_FRAME': np.nan,
                        'END_FRAME': np.nan
                    }
                final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)

            # Apply 10 Hz, 2nd-order Butterworth filter to all x/y/z landmark columns.
            final_df = butterworth_filter_df(final_df, cutoff=10, fs=50, order=2)
            final_df = calculate_angles(final_df)

            os.makedirs(os.path.join(output_folder, ath, session), exist_ok=True)
            final_df.to_csv(os.path.join(output_folder, name + '.csv'), index=False)



# Delete the videos that are not part of the Data Training Clean dataset
video_folder = 'matched_videos/'
for skipped_file in skipped:
    video_path = os.path.join(video_folder, skipped_file + '.mp4')
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f'Removed video: {video_path}')


print(f'{count-mismatch-len(skipped)} files successfully processed and included in the training dataset.')