import cv2, time, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

folder_path = 'ATH_videos_avi_processed/'
output_folder_path  = 'landing_animations/'
csv_folder_path = 'knee_angles_csv/'


def create_all_landing_animations(folder_path, csv_folder_path, output_folder_path):
    for ath in sorted(os.listdir(folder_path)):
        for session in sorted(os.listdir(os.path.join(folder_path, ath))):
            for video in sorted(os.listdir(os.path.join(folder_path, ath, session))):
                save_path = os.path.join(output_folder_path, ath, session, video.replace('_annoted.mp4', '_abduction_adduction.mp4'))
                if os.path.exists(save_path):
                    print(f'Skipping {ath} {session} {video} (already exists)')
                    continue
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f'Processing {ath} {session} {video}...')
                start_time = time.time()
                video_path = os.path.join(folder_path, ath, session, video)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                cap.release()

                window = 30
                csv_path = os.path.join(csv_folder_path, ath, session, video.replace('_annoted.mp4', '_angles.csv'))
                df = pd.read_csv(csv_path)
                mid_hip_y_list = df['mid_hip_y']
                mid_hip_y_list = pd.DataFrame(mid_hip_y_list).interpolate().bfill().values.ravel().tolist()
                left_abduction_angle_list = df['left_abduction_angle']
                left_abduction_angle_list = pd.DataFrame(left_abduction_angle_list).interpolate().bfill().values.ravel().tolist()
                mid_hip_y_derivative = np.diff(mid_hip_y_list)
                max_derivative_idx = np.argmax(mid_hip_y_derivative)
                if max_derivative_idx < window or max_derivative_idx > len(mid_hip_y_derivative) - window:
                    print('test')
                    window = min(max_derivative_idx, len(mid_hip_y_derivative) - max_derivative_idx)
                mid_hip_max = np.argmax(mid_hip_y_list[max_derivative_idx-window:max_derivative_idx+window])
                mid_hip_max_idx = max_derivative_idx - window + mid_hip_max

                plot_window = 60
                start_idx = max(0, mid_hip_max_idx - plot_window)
                end_idx = min(len(left_abduction_angle_list), mid_hip_max_idx + plot_window)

                # Ensure `angle_list` exists (computed earlier) and synchronize lengths
                try:
                    n_frames = min(len(frames), len(left_abduction_angle_list[start_idx:end_idx]))
                except NameError:
                    raise NameError('left_abduction_angle_list not found.')
                if n_frames == 0:
                    raise RuntimeError('No frames or abduction/adduction angles available to animate.')
                frames = frames[start_idx:end_idx]
                angles = np.array(left_abduction_angle_list[start_idx:end_idx][:n_frames])


                # Create figure with video on top and plot below
                fig = plt.figure(figsize=(8,8))
                fig.suptitle(f'{ath} {session} {video} Abduction/Adduction')
                ax_vid = plt.subplot2grid((3,1),(0,0), rowspan=2)
                ax_plot = plt.subplot2grid((3,1),(2,0))
                im = ax_vid.imshow(frames[0])
                ax_vid.axis('off')
                # Plot full angle trace and set up vertical frame indicator
                ax_plot.plot(np.arange(n_frames), angles, color='C0')
                vline = ax_plot.axvline(0, color='r', linewidth=2)
                ax_plot.set_xlim(0, n_frames-1)
                pad = max(5, (angles.max()-angles.min())*0.05)
                ax_plot.set_ylim(angles.min()-pad, angles.max()+pad)
                ax_plot.set_xlabel('Frame')
                ax_plot.set_ylabel('Abduction/Adduction Angle (degrees)')
                plt.tight_layout()

                # Update function: swap video frame and move vertical line
                def update(i, frames, im, vline):
                    im.set_data(frames[i])
                    vline.set_xdata([i,i])
                    return im, vline

                interval = 1000.0 / (fps if fps>0 else 30)
                ani = animation.FuncAnimation(fig, update, frames=n_frames, fargs=(frames, im, vline), blit=False, interval=interval)
                ani.save(save_path, writer='ffmpeg')
                plt.close(fig)
                end_time = time.time()
                print(f'Finished processing in {end_time - start_time:.2f} seconds.')


if __name__ == "__main__":
    create_all_landing_animations(sys.argv[1], sys.argv[2], sys.argv[3])