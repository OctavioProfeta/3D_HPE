import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, time, sys, os, pathlib, json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_pose.POSE_CONNECTIONS)
excluded_landmarks = [
    mp_pose.PoseLandmark.LEFT_EYE, 
    mp_pose.PoseLandmark.RIGHT_EYE, 
    mp_pose.PoseLandmark.LEFT_EYE_INNER, 
    mp_pose.PoseLandmark.RIGHT_EYE_INNER, 
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.RIGHT_THUMB,
    mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.LEFT_THUMB,
    mp_pose.PoseLandmark.LEFT_PINKY,
    mp_pose.PoseLandmark.LEFT_INDEX ]
for landmark in excluded_landmarks:
    # we change the way the excluded landmarks are drawn
    custom_style[landmark] = mp_drawing_styles.DrawingSpec(color=(255,255,0), thickness=None) 
    # we remove all connections which contain these landmarks
    custom_connections = [connection_tuple for connection_tuple in custom_connections if landmark.value not in connection_tuple]

pose =  mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def pose_estimation_from_folder(folder_path, output_folder_path):
    for ath in os.listdir(folder_path):
        for session in os.listdir(os.path.join(folder_path, ath)):
            session_folder_path = os.path.join(folder_path, ath, session)
            for video_name in os.listdir(session_folder_path):
                video_path = os.path.join(session_folder_path, video_name)
                output_session_folder_path = os.path.join(output_folder_path, ath, session)
                output_path = os.path.join(output_session_folder_path, pathlib.Path(video_name).stem + "_annoted" + pathlib.Path(video_name).suffix)
                cv2.startWindowThread()
                cap = cv2.VideoCapture(video_path)

                avg_fps = []

                width, height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
                video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), float(fps), (width, height))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                landmark_summary = "landmarks_summary/" + ath + "/"  + session + "/" + pathlib.Path(output_path).stem + "_results.json"           
                os.makedirs(os.path.dirname(landmark_summary), exist_ok=True)
                frames_landmarks = []
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (width, height))

                    start_time = time.time()
                    rbg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rbg_frame)
                    process_time = time.time() - start_time

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, connections=custom_connections, landmark_drawing_spec=custom_style)

                    # Collect landmarks (x, y, z, visibility) for this frame in a serializable form
                    landmarks_data = []
                    # Try common attributes that store landmarks
                    pose_landmarks = getattr(results, 'pose_world_landmarks', None)
                    if pose_landmarks is None:
                        pose_landmarks = getattr(results, 'pose_world_landmarks', None)
                    # Some result variants may store landmarks under different attributes
                    if pose_landmarks is None and hasattr(results, 'pose_results'):
                        pose_landmarks = getattr(results.pose_results, 'landmark', None)

                    if pose_landmarks is not None:
                        landmark_list = getattr(pose_landmarks, 'landmark', None) or pose_landmarks
                        if landmark_list:
                            for idx, lm in enumerate(landmark_list):
                                try:
                                    name = mp_pose.PoseLandmark(idx).name
                                    if mp_pose.PoseLandmark(idx) in excluded_landmarks:
                                        continue
                                except Exception:
                                    name = str(idx)
                                visibility = getattr(lm, 'visibility', None)
                                landmarks_data.append({
                                    'index': idx,
                                    'name': name,
                                    'x': float(lm.x),
                                    'y': float(lm.y),
                                    'z': float(lm.z),
                                    'visibility': float(visibility) if visibility is not None else None
                                })

                    frames_landmarks.append({'frame': frame_idx, 'landmarks': landmarks_data})
                    frame_idx += 1
                    
                    fps = 1 / process_time if process_time > 0 else 0
                    avg_fps.append(fps)
                    fps = sum(avg_fps) / len(avg_fps)

                    video_writer.write(frame)
                    cv2.imshow(str(), frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Write collected landmarks to JSON
                summary = {
                    'athlete': ath,
                    'session': session,
                    'video': os.path.basename(video_path),
                    'frame_count': frame_idx,
                    'frames': frames_landmarks
                }
                with open(landmark_summary, 'w') as fh:
                    json.dump(summary, fh, indent=2)

                print(f"Wrote summary to {landmark_summary}. Processed {frame_idx} frames.")


                video_writer.release()
                cap.release()

                cv2.destroyAllWindows()
                for i in range (1,5):
                    cv2.waitKey(1)

if __name__ == "__main__":
    pose_estimation_from_folder(sys.argv[1], sys.argv[2])