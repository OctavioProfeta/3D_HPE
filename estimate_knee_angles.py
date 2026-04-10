import json, os, pathlib, cv2, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    if np.cross(v1, v2)[2] > 0:  # Check the sign of the angle using the cross product
        angle = -angle
    return np.degrees(angle)


def orientation_angle(left_hip, right_hip):
    angle = np.arccos((left_hip[0]['x'] - right_hip[0]['x']) / np.sqrt((left_hip[0]['x'] - right_hip[0]['x'])**2 + (left_hip[0]['z'] - right_hip[0]['z'])**2))
    return np.degrees(angle)


def frontal_plane_normal_vector(left_hip, right_hip, left_shoulder, right_shoulder):
# This computes the normalized frontal plane normal vector using the hip and shoulder landmarks.
# It calculates the mid-point of the hips and shoulders, then creates two vectors from 
# the left hip to the mid-point and from the right hip to the mid-point.
    mid_point = [(left_hip[0]['x'] + right_hip[0]['x'] + left_shoulder[0]['x'] + right_shoulder[0]['x']) / 4, 
                 (left_hip[0]['y'] + right_hip[0]['y'] + left_shoulder[0]['y'] + right_shoulder[0]['y']) / 4, 
                 (left_hip[0]['z'] + right_hip[0]['z'] + left_shoulder[0]['z'] + right_shoulder[0]['z']) / 4]
    v1 = [left_hip[0]['x'] - mid_point[0], left_hip[0]['y'] - mid_point[1], left_hip[0]['z'] - mid_point[2]]
    v2 = [right_hip[0]['x'] - mid_point[0], right_hip[0]['y'] - mid_point[1], right_hip[0]['z'] - mid_point[2]]
    normal_vector = np.cross(v1, v2)
    return normal_vector / np.linalg.norm(normal_vector)


def normal_vector_angle(normal_vector):
    angle = np.arccos(normal_vector[2])  # Assuming the normal vector is normalized
    return np.degrees(angle)


def projection_onto_the_frontal_plane(v1, v2, n):
    v1_proj = n - np.dot(v1, n) * v1
    v2_proj = n - np.dot(v2, n) * v2
    return v1_proj, v2_proj

def compute_knee_angles(folder_path, output_folder_path):
    for ath in os.listdir(folder_path):
        for session in os.listdir(os.path.join(folder_path, ath)):
            session_folder_path = os.path.join(folder_path, ath, session)
            for json_file in os.listdir(session_folder_path):
                json_path = os.path.join(session_folder_path, json_file)
                output_session_folder_path = os.path.join(output_folder_path, ath, session)
                if not os.path.exists(output_session_folder_path):
                    os.makedirs(output_session_folder_path)
                output_path = os.path.join(output_session_folder_path, json_file.replace('_annoted_results.json', '_angles.csv'))

                try:
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                    print(f"Processing {json_path}...")
                except FileNotFoundError:
                    print(f"Error: The file {json_path} was not found.")
                
                frames_list = list(data.items())[4][1]
                normal_vector_angle_list = []
                abduction_adduction_angle_list = []
                old_angle_knee = 0
                old_angle_normal_vector = 0
                unmarked_frames = []

                for frame in frames_list:
                    if not frame['landmarks']:
                        unmarked_frames.append(frame)
                        normal_vector_angle_list.append(old_angle_normal_vector)  # Append the last known angle for unmarked frames
                        abduction_adduction_angle_list.append(old_angle_knee)  # Append the last known angle for unmarked frames
                        continue
                    else:
                        left_hip = [a for a in frame['landmarks'] if a['name'] == 'LEFT_HIP']
                        left_shoulder = [a for a in frame['landmarks'] if a['name'] == 'LEFT_SHOULDER']
                        right_hip = [a for a in frame['landmarks'] if a['name'] == 'RIGHT_HIP']
                        right_shoulder = [a for a in frame['landmarks'] if a['name'] == 'RIGHT_SHOULDER']
                        left_knee = [a for a in frame['landmarks'] if a['name'] == 'LEFT_KNEE']
                        left_ankle = [a for a in frame['landmarks'] if a['name'] == 'LEFT_ANKLE']
                        normal_vector = frontal_plane_normal_vector(left_hip, right_hip, left_shoulder, right_shoulder)
                        angle = normal_vector_angle(normal_vector)
                        femur = np.array([left_knee[0]['x'] - left_hip[0]['x'], left_knee[0]['y'] - left_hip[0]['y'], left_knee[0]['z'] - left_hip[0]['z']])
                        tibia = np.array([left_ankle[0]['x'] - left_knee[0]['x'], left_ankle[0]['y'] - left_knee[0]['y'], left_ankle[0]['z'] - left_knee[0]['z']])
                        femur_proj, tibia_proj = projection_onto_the_frontal_plane(femur, tibia, normal_vector)
                        abduction_adduction_angle = angle_between_vectors(femur_proj, tibia_proj)
                        normal_vector_angle_list.append(angle)
                        abduction_adduction_angle_list.append(abduction_adduction_angle)
                        old_angle_knee = abduction_adduction_angle
                        old_angle_normal_vector = angle
                
                # Save Abduction/Adduction angles and Normal Vector angles to a CSV file with Headers for frame_id, normal_angle, abduction_angle
                with open(output_path, 'w') as f:
                    f.write('frame_id,normal_angle,abduction_angle\n')
                    for i in range(len(frames_list)):
                        f.write(f"{frames_list[i]['frame']},{normal_vector_angle_list[i]},{abduction_adduction_angle_list[i]}\n")

    

if __name__ == "__main__":
    compute_knee_angles(sys.argv[1], sys.argv[2])