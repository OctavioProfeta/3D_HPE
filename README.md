##### A) Process the raw _'.avi'_ videos in order to get the landing animation with synchronised knee angles sliders

0. Install the required libraries listed in `requirements.txt`:
`$ conda create --name <env> --file requirements.txt`

1. The videos are organized in the following manner

    ```
        ATH_videos_avi
         └─ ATH'xx'
             └─ s'y'
                 └─ Pre Cut'zz'.avi
    ```

    where `'xx'` is the athlete number (from 1 to 25), `'y'` is the session number (1 or 2) and `'zz'` is the video number (from None to 05)

2. Run the `pose_estimation_videos.py` script to process the pose estimation on all videos:
```
$ python pose_estimation_all_videos.py ATH_videos_avi/ ATH_videos_avi_processed/
```
This process will save the videos with the drawn landmarks in `ATH_videos_avi_processed` and create a new folder `landmarks_summary` which contains `.json` files with all the data.

3. Run the `estimate_knee_angles.py` script to process all the abduction/adduction and flexion/extension knee angles into a `.csv`file:
```
$ python estimate_knee_angles.py landmarks_summary/ATH_videos_avi/ knee_angles_csv/.
```
This process will save the `.csv` files with all the calculated knee angles into the `knee_angles_csv` folder.

4. Run the `create_all_landing_animation.py` script to process all the landing animation with the synchronised knee angles sliders:
```
$ python create_all_landing_animation.py ATH_videos_avi_processed/ knee_angles_csv/ landing_animations/
```
This process will save all the landing animation in the `landing_animation/` folder.

5. **(Optional)** You can run the `plot_all_*.py` script if you want to plot the evolution of certain angles or the mid_hip_y_value over all frames of the videos:
```
$ python plot_all_*.py knee_angles_csv/ <folder name>
```
This process will save all figures to `<folder name>`.

#### B) Get the ground truth and synchronised data for training

0. Run the `pose_estimation_videos.py` script if not done already. There should be a `landmarks_summary` folder which contains `.json` files with all the data.
Download all the `Subjects Data/` folder and put the folder in the current directory.

1. Run the `gt_extraction.py` script to extract the files we are interested in:
```
$ python gt_extraction.py
```
This will create the folder `ACL_Lengths/` which contains all our ground truth.

2. Run the `create_matched_data.py` script to match the ground truth to the videos and the pose estimation.
```
$ python create_matched_data.py landmarks_summary/
```
This will create 4 new folders containing the matched data.

3. **(Optional)** Run the `create_all_acl_strain_animation.py`script to get the animation of the landing with the ACL strain on both sides.

 