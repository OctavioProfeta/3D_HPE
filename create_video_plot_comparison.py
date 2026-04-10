# Synchronized video (top) + knee-angle plot (bottom) animation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


video_path = 'test_annoted.avi'
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

# Ensure `angle_list` exists (computed earlier) and synchronize lengths
try:
    n_frames = min(len(frames), len(abduction_adduction_angle_list))
except NameError:
    raise NameError('abduction_adduction_angle_list not found. Run the cell that computes abduction and adduction angles first.')
if n_frames == 0:
    raise RuntimeError('No frames or abduction/adduction angles available to animate.')
frames = frames[:n_frames]
angles = np.array(abduction_adduction_angle_list[:n_frames])
# vectors = np.array(orientation_list[:n_frames])

# Create figure with video on top and plot below
fig = plt.figure(figsize=(8,8))
ax_vid = plt.subplot2grid((3,1),(0,0), rowspan=2)
ax_plot = plt.subplot2grid((3,1),(2,0))
im = ax_vid.imshow(frames[0])
ax_vid.axis('off')
# Plot full angle trace and set up vertical frame indicator
ax_plot.plot(np.arange(n_frames), angles, color='C0')
# ax_plot.plot(np.arange(n_frames), vectors, color='C1')
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
# Display as JS animation inline in the notebook
ani.save('abduction_adduction.gif', writer='ffmpeg')