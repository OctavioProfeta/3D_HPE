import json, os, sys
import numpy as np


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
    mid_point = np.array([(left_hip[0]['x'] + right_hip[0]['x'] + left_shoulder[0]['x'] + right_shoulder[0]['x']) / 4, 
                         (left_hip[0]['y'] + right_hip[0]['y'] + left_shoulder[0]['y'] + right_shoulder[0]['y']) / 4, 
                         (left_hip[0]['z'] + right_hip[0]['z'] + left_shoulder[0]['z'] + right_shoulder[0]['z']) / 4])
    v1 = np.array([left_hip[0]['x'] - mid_point[0], left_hip[0]['y'] - mid_point[1], left_hip[0]['z'] - mid_point[2]])
    v2 = np.array([right_hip[0]['x'] - mid_point[0], right_hip[0]['y'] - mid_point[1], right_hip[0]['z'] - mid_point[2]])
    normal_vector = np.cross(v1, v2)
    return normal_vector / np.linalg.norm(normal_vector)


def sagittal_plane_normal_vector(left_hip, right_hip):
# This compute the normalized sagittal plane normal vector using the hip landmarks. 
# It creates a vector from the right hip to the left hip, which is perpendicular to the sagittal plane.
    normal_vector = np.array([left_hip[0]['x'] - right_hip[0]['x'], left_hip[0]['y'] - right_hip[0]['y'], left_hip[0]['z'] - right_hip[0]['z']])
    return normal_vector / np.linalg.norm(normal_vector)


def normal_vector_angle(normal_vector):
    angle = np.arccos(normal_vector[2])  # Assuming the normal vector is normalized
    return np.degrees(angle)


def projection_onto_plane(v1, v2, n):
# This would normaly need to pass a point on the plane, but since both planes passes through the world coordinate
# origin (mid_hip point), we don't need to pass it as an argument.
    v1_proj = v1 - np.dot(v1, n) * n / np.dot(n, n)
    v2_proj = v2 - np.dot(v2, n) * n / np.dot(n, n)
    return v1_proj, v2_proj


def compute_knee_angles(folder_path, output_folder_path):
    for ath in sorted(os.listdir(folder_path)):
        for session in sorted(os.listdir(os.path.join(folder_path, ath))):
            session_folder_path = os.path.join(folder_path, ath, session)
            for json_file in sorted(os.listdir(session_folder_path)):
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
                frontal_plane_vector_list = []
                sagittal_plane_vector_list = []
                left_abduction_adduction_angle_list = []
                right_abduction_adduction_angle_list = []
                left_flexion_extension_angle_list = []
                right_flexion_extension_angle_list = []
                mid_hip_y_frame_list = []
                left_abduction_adduction_old_angle = 0
                right_abduction_adduction_old_angle = 0
                left_flexion_extension_old_angle = 0
                right_flexion_extension_old_angle = 0
                frontal_plane_old_angle = 0
                sagittal_plane_old_angle = 0
                mid_hip_y_frame_old_value = 0
                unmarked_frames = []

                for frame in frames_list:
                    if not frame['landmarks']:
                        unmarked_frames.append(frame)
                        frontal_plane_vector_list.append(frontal_plane_old_angle)  # Append the last known angle for unmarked frames
                        sagittal_plane_vector_list.append(sagittal_plane_old_angle)  # Append the last known angle for unmarked frames
                        left_abduction_adduction_angle_list.append(left_abduction_adduction_old_angle)  # Append the last known angle for unmarked frames
                        right_abduction_adduction_angle_list.append(right_abduction_adduction_old_angle)  # Append the last known angle for unmarked frames
                        left_flexion_extension_angle_list.append(left_flexion_extension_old_angle)  # Append the last known angle for unmarked frames
                        right_flexion_extension_angle_list.append(right_flexion_extension_old_angle)  # Append the last known angle for unmarked frames
                        mid_hip_y_frame_list.append(np.nan) # Append NaN for unmarked frames (we later extrapolate)
                        continue
                    else:
                        left_hip = [a for a in frame['landmarks'] if a['name'] == 'LEFT_HIP']
                        left_shoulder = [a for a in frame['landmarks'] if a['name'] == 'LEFT_SHOULDER']
                        right_hip = [a for a in frame['landmarks'] if a['name'] == 'RIGHT_HIP']
                        right_shoulder = [a for a in frame['landmarks'] if a['name'] == 'RIGHT_SHOULDER']
                        left_knee = [a for a in frame['landmarks'] if a['name'] == 'LEFT_KNEE']
                        left_ankle = [a for a in frame['landmarks'] if a['name'] == 'LEFT_ANKLE']
                        right_knee = [a for a in frame['landmarks'] if a['name'] == 'RIGHT_KNEE']
                        right_ankle = [a for a in frame['landmarks'] if a['name'] == 'RIGHT_ANKLE']
                        right_hip_frame = [a for a in frame['landmarks'] if a['name'] == 'RIGHT_HIP_frame_reference']
                        left_hip_frame = [a for a in frame['landmarks'] if a['name'] == 'LEFT_HIP_frame_reference']

                        mid_hip_y_frame = right_hip_frame[0]['y'] + left_hip_frame[0]['y'] / 2
                        mid_hip_y_frame_list.append(mid_hip_y_frame)

                        frontal_vector = frontal_plane_normal_vector(left_hip, right_hip, left_shoulder, right_shoulder)
                        sagittal_vector = sagittal_plane_normal_vector(left_hip, right_hip)
                        frontal_plane_angle = normal_vector_angle(frontal_vector)
                        sagittal_plane_angle = normal_vector_angle(sagittal_vector)

                        left_femur = np.array([left_knee[0]['x'] - left_hip[0]['x'], left_knee[0]['y'] - left_hip[0]['y'], left_knee[0]['z'] - left_hip[0]['z']])
                        left_tibia = np.array([left_ankle[0]['x'] - left_knee[0]['x'], left_ankle[0]['y'] - left_knee[0]['y'], left_ankle[0]['z'] - left_knee[0]['z']])
                        right_femur = np.array([right_knee[0]['x'] - right_hip[0]['x'], right_knee[0]['y'] - right_hip[0]['y'], right_knee[0]['z'] - right_hip[0]['z']])
                        right_tibia = np.array([right_ankle[0]['x'] - right_knee[0]['x'], right_ankle[0]['y'] - right_knee[0]['y'], right_ankle[0]['z'] - right_knee[0]['z']])

                        left_femur_proj_frontal, left_tibia_proj_frontal = projection_onto_plane(left_femur, left_tibia, frontal_vector)
                        right_femur_proj_frontal, right_tibia_proj_frontal = projection_onto_plane(right_femur, right_tibia, frontal_vector)
                        left_abduction_adduction_angle = angle_between_vectors(left_femur_proj_frontal, left_tibia_proj_frontal)
                        right_abduction_adduction_angle = angle_between_vectors(right_femur_proj_frontal, right_tibia_proj_frontal)

                        left_femur_proj_sagittal, left_tibia_proj_sagittal = projection_onto_plane(left_femur, left_tibia, sagittal_vector)
                        right_femur_proj_sagittal, right_tibia_proj_sagittal = projection_onto_plane(right_femur, right_tibia, sagittal_vector)
                        left_flexion_extension_angle = angle_between_vectors(left_femur_proj_sagittal, left_tibia_proj_sagittal)
                        right_flexion_extension_angle = angle_between_vectors(right_femur_proj_sagittal, right_tibia_proj_sagittal)

                        frontal_plane_vector_list.append(frontal_plane_angle)
                        sagittal_plane_vector_list.append(sagittal_plane_angle)
                        left_abduction_adduction_angle_list.append(left_abduction_adduction_angle)
                        right_abduction_adduction_angle_list.append(right_abduction_adduction_angle)
                        left_flexion_extension_angle_list.append(left_flexion_extension_angle)
                        right_flexion_extension_angle_list.append(right_flexion_extension_angle)

                        frontal_plane_old_angle = frontal_plane_angle
                        left_abduction_adduction_old_angle = left_abduction_adduction_angle
                        right_abduction_adduction_old_angle = right_abduction_adduction_angle
                        left_flexion_extension_old_angle = left_flexion_extension_angle
                        right_flexion_extension_old_angle = right_flexion_extension_angle
                
                # Save Abduction/Adduction angles and Normal Vector angles to a CSV file with Headers for frame_id, normal_angle, abduction_angle
                with open(output_path, 'w') as f:
                    f.write('frame_id,frontal_plane_angle,sagittal_plane_angle,left_abduction_angle,right_abduction_angle,left_flexion_angle,right_flexion_angle,mid_hip_y\n')
                    for i in range(len(frames_list)):
                        f.write(f"{frames_list[i]['frame']},{frontal_plane_vector_list[i]},{sagittal_plane_vector_list[i]},{left_abduction_adduction_angle_list[i]},{right_abduction_adduction_angle_list[i]},{left_flexion_extension_angle_list[i]},{right_flexion_extension_angle_list[i]},{mid_hip_y_frame_list[i]}\n")

    

if __name__ == "__main__":
    compute_knee_angles(sys.argv[1], sys.argv[2])