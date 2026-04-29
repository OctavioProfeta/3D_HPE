import pandas as pd
import os, json, shutil, sys


sto_folder = 'ACL_Lengths/'

new_json_folder = 'matched_json/'
new_sto_folder = 'matched_acl_length/'
new_video_folder = 'matched_videos/'
new_csv_folder = 'matched_csv/'

def create_matched_data(json_folder):
    mismatch = 0
    count = 0
    for ath in sorted(os.listdir(json_folder)):
        for session in sorted(os.listdir(os.path.join(json_folder, ath))):
            for json_file in sorted(os.listdir(os.path.join(json_folder, ath, session))):
                count += 1
                json_path = os.path.join(json_folder, ath, session, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                frame_count = data['frame_count']
                name = json_file[:9].replace(' ', '')
                if name[6] == '.':
                    name = name[:6]
                name = name.replace('0', '')
                name = name.replace('.', '')
                sto_file = os.path.join(sto_folder, ath, session, name + '_MuscleAnalysis_Length.sto')
                sto = pd.read_csv(sto_file, sep=r"\s+", skiprows=11)
                if abs(len(sto)/2 - frame_count) > 2:
                    print(f'{ath}/{session}/{name}: frame_count={frame_count}, sto_frames={len(sto)/2}')
                    mismatch += 1
                    continue
                os.makedirs(os.path.join(new_json_folder, ath, session), exist_ok=True)
                os.makedirs(os.path.join(new_sto_folder, ath, session), exist_ok=True)
                os.makedirs(os.path.join(new_video_folder, ath, session), exist_ok=True)
                os.makedirs(os.path.join(new_csv_folder, ath, session), exist_ok=True)
                shutil.copy(json_path, os.path.join(new_json_folder, ath, session, name + '.json'))
                shutil.copy(sto_file, os.path.join(new_sto_folder, ath, session, name + '.sto'))
                video_file = os.path.join('ATH_videos_avi_processed', ath, session, json_file.replace('_results.json', '.mp4'))
                shutil.copy(video_file, os.path.join(new_video_folder, ath, session, name + '.mp4'))
                csv_file = os.path.join('knee_angles_csv', ath, session, json_file.replace('_annoted_results.json', '_angles.csv'))
                shutil.copy(csv_file, os.path.join(new_csv_folder, ath, session, name + '.csv'))
    print(f'Total files processed: {count}')
    print(f'Total mismatches: {mismatch}')
    print(f'Percentage of mismatches: {mismatch/count*100:.2f}%')
    
if __name__ == "__main__":
    create_matched_data(sys.argv[1])