# A) Process the raw _'.avi'_ videos in order to get the landmarks later used in the ACL strain estimation

#### On mac, run `find . -name ".DS_Store" -type f -delete` before anything to delete all `.DS_Store` files from folders and subfolders

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

2. Run the `pose_estimation_all_videos.py` script to process the pose estimation on all videos:
```
$ python pose_estimation_all_videos.py ATH_videos_avi/ ATH_videos_avi_processed/
```
This process will save the videos with the drawn landmarks in `ATH_videos_avi_processed` and create a new folder `landmarks_summary` which contains `.json` files with all the data.

# B) Get the ground truth and synchronised data for training

0. Run the `pose_estimation_videos.py` script if not done already. There should be a `landmarks_summary` folder which contains `.json` files with all the data.
Download all the `Subjects Data/` folder and put the folder in the current directory.

1. Run the `gt_extraction.py` script to extract the files we are interested in:
```
$ python gt_extraction.py
```
This will create the folder `ACL_Lengths/` which contains all our ground truth.


2. Run the `create_training_data.py` to create the folder `Data Training Clean/` folder containing the data used to train the models.

````
python create_training_data.py
````


3. **(Optional)** Run the `create_all_acl_strain_animation.py`script to get the animation of the landing with the ACL strain on both sides.

# C) Train the model

1. The `bi_lstm.ipynb` notebook contains everythig needed for training.
 