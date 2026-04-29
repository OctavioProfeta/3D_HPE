import cv2, time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

folder_path = 'matched_videos/'
output_folder_path  = 'matched_landing/'
csv_folder_path = 'matched_csv/'
sto_folder_path = 'matched_acl_length/'

for ath in sorted(os.listdir(folder_path)):
    for session in sorted(os.listdir(os.path.join(folder_path, ath))):
        for video in sorted(os.listdir(os.path.join(folder_path, ath, session))):
            name = os.path.splitext(video)[0]
            save_path = os.path.join(output_folder_path, ath, session, name + '_animation.mp4')
            if os.path.exists(save_path):
                print(f'Skipping {ath} {session} {video} (already exists)')
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f'Processing {ath} {session} {video}...')
            start_time = time.time()
            video_path = os.path.join(folder_path, ath, session, video)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            window = 30
            csv_path = os.path.join(csv_folder_path, ath, session, name + '.csv')
            df = pd.read_csv(csv_path)
            mid_hip_y_list = df['mid_hip_y']
            mid_hip_y_list = pd.DataFrame(mid_hip_y_list).interpolate().bfill().values.ravel().tolist()
            
            mid_hip_y_derivative = np.diff(mid_hip_y_list)
            max_derivative_idx = np.argmax(mid_hip_y_derivative)
            

            plot_window = int(fps)
            start_idx = max(0, max_derivative_idx - plot_window)
            end_idx = min(len(frames), max_derivative_idx + plot_window)

            # Load the ACL lenght data
            sto_file = os.path.join(sto_folder_path, ath, session, name + '.sto')
            sto = pd.read_csv(sto_file, sep=r"\s+", skiprows=11)
            sto = iqr_roll_median(sto, window=5, k=1)

            l_zero_idx = start_idx - 20
            if l_zero_idx % 2 != 0:
                l_zero_idx -= 1

            frames = frames[start_idx:end_idx]
            right_acl_length = sto['acl_r']
            right_acl_length = right_acl_length[::2]
            right_acl_strain = 1 + (right_acl_length - right_acl_length[l_zero_idx])/right_acl_length[l_zero_idx]
            right_acl_strain = right_acl_strain[start_idx:end_idx].tolist()
            left_acl_length = sto['acl_l']
            left_acl_length = left_acl_length[::2]
            left_acl_strain = 1 + (left_acl_length - left_acl_length[l_zero_idx])/left_acl_length[l_zero_idx]
            left_acl_strain = left_acl_strain[start_idx:end_idx].tolist()

            n_frames = min(len(frames), len(right_acl_length), len(left_acl_length))

            # Create figure with video on top and 2 plots below
            fig, (ax_vid, ax_plot, ax_plot_2) = plt.subplots(
                3,
                1,
                figsize=(8, 10),
                gridspec_kw={'height_ratios': [3, 1, 1]}
            )
            fig.suptitle(f'{ath} {session} {video} ACL Strain')
            im = ax_vid.imshow(frames[0])
            ax_vid.axis('off')
            # Plot abduction/adduction angle trace and set up vertical frame indicator
            ax_plot.plot(np.arange(n_frames), left_acl_strain, color='C1', label='Left ACL Strain')
            vline = ax_plot.axvline(0, color='r', linewidth=2)
            ax_plot.set_xlim(0, n_frames-1)
            # pad1 = max(5, (max(left_acl_length)-min(left_acl_length))*0.05)
            # ax_plot.set_ylim(min(left_acl_length)-pad1, max(left_acl_length)+pad1)
            ax_plot.set_xlim(0, n_frames-1)
            # Set the xticks to be start_idx, max_derivative_idx, and end_idx (relative to the plot window)
            xticks = [0, max_derivative_idx - start_idx, end_idx - start_idx - 1]
            xtick_labels = [str(start_idx), str(max_derivative_idx), str(end_idx)]
            ax_plot.set_xticks(xticks)
            ax_plot.set_xticklabels(xtick_labels)
            ax_plot.set_xlabel('Frame')
            ax_plot.set_ylabel('Left ACL Strain')
            ax_plot.legend()
            # Plot flexion/extension angle trace and set up vertical frame indicator
            ax_plot_2.plot(np.arange(n_frames), right_acl_strain, color='C0', label='Right ACL Strain ')
            vline_2 = ax_plot_2.axvline(0, color='r', linewidth=2)
            # pad2 = max(5, (max(right_acl_length)-min(right_acl_length))*0.05)
            # ax_plot_2.set_ylim(min(right_acl_length)-pad2, max(right_acl_length)+pad2)
            ax_plot_2.set_xlim(0, n_frames-1)
            # Set the xticks to be start_idx, max_derivative_idx, and end_idx (relative to the plot window)
            ax_plot_2.set_xticks(xticks)
            ax_plot_2.set_xticklabels(xtick_labels)
            ax_plot_2.set_xlabel('Frame')
            ax_plot_2.set_ylabel('Right ACL Strain')
            ax_plot_2.legend()
            plt.tight_layout()

            # Update function: swap video frame and move vertical line
            def update(i, frames, im, vline, vline_2):
                im.set_data(frames[i])
                vline.set_xdata([i,i])
                vline_2.set_xdata([i,i])
                return im, vline, vline_2

            interval = 1000.0 / (fps if fps>0 else 30)
            ani = animation.FuncAnimation(fig, update, frames=n_frames, fargs=(frames, im, vline, vline_2), blit=False, interval=interval)
            ani.save(save_path, writer='ffmpeg')
            plt.close(fig)
            end_time = time.time()
            print(f'Finished processing in {end_time - start_time:.2f} seconds.')