import cv2
import sys
sys.path.append("../..")
from MovenetDepthai import MovenetDepthai, KEYPOINT_DICT
from MovenetRenderer import MovenetRenderer
import argparse
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['lightning', 'thunder'], default='thunder',
                        help="Model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
args = parser.parse_args()            
df = pd.DataFrame(columns=['timestamp',
    'nose','left_eye','right_eye','left_ear',
    'right_ear','left_shoulder','right_shoulder','left_elbow',
    'right_elbow','left_wrist','right_wrist',
    'left_hip','right_hip','left_knee',
    'right_knee','left_ankle','right_ankle'
])
data = []
pose = MovenetDepthai(input_src=args.input, model=args.model)
renderer = MovenetRenderer(pose, output=args.output)


while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: 
        break
    timestamp = time.time()
    row = {'timestamp': timestamp, 
           'nose': body.keypoints_norm[0], 
           'left_eye': body.keypoints_norm[1], 
           'right_eye': body.keypoints_norm[2], 
           'left_ear': body.keypoints_norm[3], 
           'right_ear': body.keypoints_norm[4], 
           'left_shoulder': body.keypoints_norm[5], 
           'right_shoulder': body.keypoints_norm[6], 
           'left_elbow': body.keypoints_norm[7], 
           'right_elbow': body.keypoints_norm[8], 
           'left_wrist': body.keypoints_norm[9], 
           'right_wrist': body.keypoints_norm[10], 
           'left_hip': body.keypoints_norm[11], 
           'right_hip': body.keypoints_norm[12], 
           'left_knee': body.keypoints_norm[13], 
           'right_knee': body.keypoints_norm[14], 
           'left_ankle': body.keypoints_norm[15], 
           'right_ankle': body.keypoints_norm[16]}
    data.append(row)
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    with pd.ExcelWriter(r'C:\Users\Uporabnik\Desktop\testing.xlsx') as writer: 
        df.to_excel(writer, sheet_name='df_1') 
    # Write data to Excel file
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

renderer.exit()
pose.exit()

