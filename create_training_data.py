import json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        upper = Q3[col] - k * IQR[col]

        outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
        rolling_med = df_clean[col].rolling(window=window, center=True, min_periods=1).median()

        df_clean.loc[outliers, col] = rolling_med[outliers]
        
    return df_clean

csv_folder = 'matched_csv'
json_folder = 'matched_json'
acl_folder = 'matched_acl_length'
mot_folder = 'matched_mot'

output_folder = 'Data Training Clean/'

skipped = []

for ath in sorted(os.listdir(csv_folder)):
    for session in sorted(os.listdir(os.path.join(csv_folder, ath))):
        for trial in sorted(os.listdir(os.path.join(csv_folder, ath, session))):
            print(f'Processing {ath} {session} {trial}...')
            name = f'{ath}/{session}/{trial.replace(".csv", "")}'
            # Load CSV data
            csv_path = os.path.join(csv_folder, ath, session, trial)
            csv_df = pd.read_csv(csv_path)
            
            # Load JSON data
            json_path = os.path.join(json_folder, name + '.json')
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # Find the 100 frames we'll be using around mid hip max derivative
            # mid_hip_y_list = pd.DataFrame(csv_df['mid_hip_y']).interpolate().bfill().values.ravel().tolist()
            # mid_hip_y_derivative = np.diff(mid_hip_y_list)
            # max_derivative_idx = np.argmax(mid_hip_y_derivative)
            # plot_window = 50 # This is the FPS of the videos, so this will give us 1 second before and after the max derivative
            # start_idx = max(0, max_derivative_idx - plot_window)
            # end_idx = min(len(mid_hip_y_list), max_derivative_idx + plot_window)

            # Find the indices of where the force on the plates are non null
            mot_file = os.path.join(mot_folder, name + '.mot')
            mot = pd.read_csv(mot_file, sep=r"\s+", skiprows=5)
            condition = (mot["ground_force_px"] > 0) | (mot["1_ground_force_px"] > 0)
            true_indices = np.where(condition)[0] / 2
            start_idx = int(min(true_indices) - 20)
            end_idx = int(max(true_indices) + 20)

            json_frames = json_data['frames'][start_idx:end_idx]
            
            
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

            if max(right_acl_strain) > 1.35 or min(right_acl_strain) < 0.65 or max(left_acl_strain) > 1.35 or min(left_acl_strain) < 0.65:
                print(f'Warning: Unusual ACL strain values for {ath} {session} {trial}')
                skipped.append(name)
                continue

            csv_df = csv_df[start_idx:end_idx].reset_index(drop=True)

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
                        'LEFT_ABDUCTION_ANGLE': csv_df.loc[i, 'left_abduction_angle'],
                        'RIGHT_ABDUCTION_ANGLE': csv_df.loc[i, 'right_abduction_angle'],
                        'LEFT_FLEXION_ANGLE': csv_df.loc[i, 'left_flexion_angle'],
                        'RIGHT_FLEXION_ANGLE': csv_df.loc[i, 'right_flexion_angle'],
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
                        'LEFT_ABDUCTION_ANGLE': np.nan,
                        'RIGHT_ABDUCTION_ANGLE': np.nan,
                        'LEFT_FLEXION_ANGLE': np.nan,
                        'RIGHT_FLEXION_ANGLE': np.nan,
                        'RIGHT_ACL_STRAIN': np.nan,
                        'LEFT_ACL_STRAIN': np.nan,
                        'START_FRAME': np.nan,
                        'END_FRAME': np.nan
                    }
                final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)
            os.makedirs(os.path.join(output_folder, ath, session), exist_ok=True)
            final_df.to_csv(os.path.join(output_folder, name + '.csv'), index=False)
    

# Delete the videos that are not part of the Data Training Clean dataset
video_folder = 'matched_videos/'
for skipped_file in skipped:
    video_path = os.path.join(video_folder, skipped_file + '.mp4')
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f'Removed video: {video_path}')


# Make a plot with all ACL lenghts for all trials to see if there are any outliers we should remove
import matplotlib.pyplot as plt
import os
import pandas as pd
acl_folder = 'Data Training Clean/'
plot_folder = 'ACL Plots Checks/'
for ath in sorted(os.listdir(acl_folder)):
    for session in sorted(os.listdir(os.path.join(acl_folder, ath))):
        # We plot all trials for a session onto 6 subplots, one for each trial
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{ath} {session} ACL Strain')
        for i, trial in enumerate(sorted(os.listdir(os.path.join(acl_folder, ath, session)))):
            trial_path = os.path.join(acl_folder, ath, session, trial)
            df = pd.read_csv(trial_path)
            axs[i//3, i%3].plot(df['RIGHT_ACL_STRAIN'], label='Right ACL Strain')
            axs[i//3, i%3].plot(df['LEFT_ACL_STRAIN'], label='Left ACL Strain')
            axs[i//3, i%3].set_title(f'Trial {trial.replace(".csv", "")}')
            axs[i//3, i%3].legend()
        plt.tight_layout()
        save_path = os.path.join(plot_folder, ath, f'{session}_acl_strain.png')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(os.path.join(save_path))
        plt.close()