# Extract all PreCut(i)_MuscleAnalysis_Length.sto files from Subjects Data/ATH(j)/s(k)/Results/PreCut(i)_MuscleAnalysis/ and save them in
# a new directory called ACL_Lengths/ATH(j)/s(k)/PreCut(i)_ACL_Length/
import os
import shutil
# Define the base directory where the Subjects Data is located
base_dir = 'Subjects Data'
# Define the new base directory where the ACL_Lengths will be saved
new_base_dir = 'ACL_Lengths'
# Loop through each subject in the base directory
for subject in sorted(os.listdir(base_dir)):
    subject_path = os.path.join(base_dir, subject)
    if os.path.isdir(subject_path):
        # Loop through each session in the subject directory
        for session in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session)
            if os.path.isdir(session_path):
                precut_path = os.path.join(session_path, 'Results')
                for muscle_analysis_folder in sorted(os.listdir(precut_path)):
                    if os.path.isdir(os.path.join(precut_path, muscle_analysis_folder)):
                        muscle_analysis_path = os.path.join(precut_path, muscle_analysis_folder)
                        # Look for the Length.sto file in the MuscleAnalysis directory
                        for file in os.listdir(muscle_analysis_path):
                            if file.endswith('_Length.sto'):
                                length_file_path = os.path.join(muscle_analysis_path, file)
                                # Define the new directory path for saving the Length.sto file
                                new_dir = os.path.join(new_base_dir, subject[:5], session)                                
                                os.makedirs(new_dir, exist_ok=True)
                                # Define the new file path
                                new_file_name = file.replace(file[6], str(int(file[6])-1))
                                if new_file_name[6] == '0':
                                    new_file_name = new_file_name.replace(new_file_name[6], '')
                                new_file_path = os.path.join(new_dir, new_file_name)
                                # Copy the Length.sto file to the new location
                                shutil.copy(length_file_path, new_file_path)
                                print(f'Copied {length_file_path} to {new_file_path}')