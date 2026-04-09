import json
import numpy as np
import matplotlib.pyplot as plt

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