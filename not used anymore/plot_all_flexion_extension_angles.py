import pandas as pd
import matplotlib.pyplot as plt
import os, sys

def plot_all_flexion_extension_angles(input_folder, output_folder):
    for ath in sorted(os.listdir(input_folder)):
        for session in sorted(os.listdir(os.path.join(input_folder, ath))):
            print(f'Processing {ath} {session}...')
            for side in ['left', 'right']:
                knee = side + '_flexion_angle'
                session_folder_path = os.path.join(input_folder, ath, session)
                # Make a subplot that will plot all 6 csv files in the session folder
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Flexion/Extension Angle Over Frames for {ath} {session}')
                axes = axes.flatten()
                i = 0
                for csv_file in sorted(os.listdir(session_folder_path)):
                    csv_path = os.path.join(session_folder_path, csv_file)
                    df = pd.read_csv(csv_path)
                    axes[i].plot(df[knee], label=f'{side.capitalize()} Knee Flexion/Extension Angle')
                    axes[i].set_xlabel('Frame')
                    axes[i].set_ylabel('Flexion/Extension Angle (degrees)')
                    axes[i].set_title(f'{csv_file}')
                    axes[i].legend()
                    i += 1
                plt.tight_layout()
                if not os.path.exists(f'{output_folder}/{ath}'):
                    os.makedirs(f'{output_folder}/{ath}')
                plt.savefig(f'{output_folder}/{ath}/{session}_{side}_flexion_extension_angles.png')
                plt.close()


if __name__ == "__main__":
    plot_all_flexion_extension_angles(sys.argv[1], sys.argv[2])