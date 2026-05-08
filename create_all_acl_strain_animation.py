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
csv_folder_path = 'Data Training Clean/'

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

            csv_path = os.path.join(csv_folder_path, ath, session, name + '.csv')
            df = pd.read_csv(csv_path)
            left_acl_strain = df['LEFT_ACL_STRAIN'].values
            right_acl_strain = df['RIGHT_ACL_STRAIN'].values
            start_idx = int(df['START_FRAME'].mode().iloc[0])
            end_idx = int(df['END_FRAME'].mode().iloc[0])

            # Get the frames [start_idx, end_idx] from the video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            for i in range(end_idx+1):
                ret, frame = cap.read()
                if not ret:
                    break
                if i >= start_idx:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()


            n_frames = min(len(frames), len(right_acl_strain), len(left_acl_strain))

            # Create figure with video on top and 2 plots below
            fig, (ax_vid, ax_plot, ax_plot_2) = plt.subplots(
                3,
                1,
                figsize=(8, 10),
                gridspec_kw={'height_ratios': [3, 1, 1]}
            )
            fig.suptitle(f'{ath} {session} {video} Abduction/Adduction')
            im = ax_vid.imshow(frames[0])
            ax_vid.axis('off')
            # Plot abduction/adduction angle trace and set up vertical frame indicator
            ax_plot.plot(np.arange(n_frames), left_acl_strain, color='C1', label='Left ACL Strain')
            vline = ax_plot.axvline(start_idx, color='r', linewidth=2)
            ax_plot.set_xlim(0, n_frames-1)
            # pad1 = max(5, (max(left_acl_length)-min(left_acl_length))*0.05)
            # ax_plot.set_ylim(min(left_acl_length)-pad1, max(left_acl_length)+pad1)
            ax_plot.set_xlim(0, n_frames-1)
            ax_plot.set_xlabel('Frame')
            ax_plot.set_ylabel('Left ACL Strain')
            # Set the x tick label to be start and end frame numbers (0 == start_idx, max == end_idx)
            ax_plot.set_xticks([0, n_frames-1])
            ax_plot.set_xticklabels([start_idx, end_idx])
            ax_plot.legend()
            # Plot flexion/extension angle trace and set up vertical frame indicator
            ax_plot_2.plot(np.arange(n_frames), right_acl_strain, color='C0', label='Right ACL Strain ')
            vline_2 = ax_plot_2.axvline(start_idx, color='r', linewidth=2)
            # pad2 = max(5, (max(right_acl_length)-min(right_acl_length))*0.05)
            # ax_plot_2.set_ylim(min(right_acl_length)-pad2, max(right_acl_length)+pad2)
            ax_plot_2.set_xlim(0, n_frames-1)
            ax_plot_2.set_xlabel('Frame')
            ax_plot_2.set_ylabel('Right ACL Strain')
            ax_plot_2.set_xticks([0, n_frames-1])
            ax_plot_2.set_xticklabels([start_idx, end_idx])
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