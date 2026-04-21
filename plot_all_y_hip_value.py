import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

def plot_all_y_hip_values(folder_path, output_folder_path):
    for ath in sorted(os.listdir(folder_path)):
        for session in sorted(os.listdir(os.path.join(folder_path, ath))):
            session_folder_path = os.path.join(folder_path, ath, session)
            print(f'Processing {ath} {session}...')
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Mid Hip Y Coordinate Over Frames for {ath} {session}')
            axes = axes.flatten()
            i = 0
            for csv_file in sorted(os.listdir(session_folder_path)):
                window = 30
                csv_path = os.path.join(session_folder_path, csv_file)
                df = pd.read_csv(csv_path)
                mid_hip_y_list = df['mid_hip_y']
                mid_hip_y_list = pd.DataFrame(mid_hip_y_list).interpolate().bfill().values.ravel().tolist()
                mid_hip_y_derivative = np.diff(mid_hip_y_list)
                max_derivative_idx = np.argmax(mid_hip_y_derivative)
                if max_derivative_idx < window or max_derivative_idx > len(mid_hip_y_derivative) - window:
                    window = min(max_derivative_idx, len(mid_hip_y_derivative) - max_derivative_idx)
                mid_hip_max = np.argmax(mid_hip_y_list[max_derivative_idx-window:max_derivative_idx+window])

                # Plot mid hip y coordinate and its derivative
                axes[i].plot(mid_hip_y_list, label='Mid Hip Y Coordinate')
                axes[i].scatter(max_derivative_idx, mid_hip_y_list[max_derivative_idx], color='red', label='Max Derivative')
                axes[i].scatter(max_derivative_idx-window+mid_hip_max, mid_hip_y_list[max_derivative_idx-window+mid_hip_max], color='green', label='Max Mid Hip')
                axes[i].set_title(f'{csv_file}')
                axes[i].legend()
                i += 1
                
            plt.tight_layout()
            # plt.show()
            if not os.path.exists(f'{output_folder_path}/{ath}'):
                os.makedirs(f'{output_folder_path}/{ath}')
            plt.savefig(f'{output_folder_path}/{ath}/{session}_mid_hip_y_values.png')
            plt.close()


if __name__ == "__main__":
    plot_all_y_hip_values(sys.argv[1], sys.argv[2])