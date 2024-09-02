
import cv2
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd


csv_path = "Final.csv"
vid_dir = "videos"
output_dir = "processed_videos"
frame_rate = 30
vid_dim = (1280, 720)

def crop_image(frame, insta = False, s = ''):
    if insta and s!='':
        s_res, e_res = [float(x.strip()) for x in s.split(',')]
        # print(s_res, e_res)
        return frame[int(frame.shape[0]*s_res):-1*int(frame.shape[0]*e_res)]
    return frame


if __name__ == '__main__':
    df = pd.read_csv(csv_path)
    for i in range(len(df)):
            print(f"{i}    {df['File Name'][i]}_{df['Counter'][i]}")   
            path = vid_dir + f"\\{int(df['File Name'][i])}_{int(df['Counter'][i])}.mp4"
            cap = cv2.VideoCapture(path)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v'  )
            out = cv2.VideoWriter(output_dir, fourcc, frame_rate, vid_dim)
            insta_story = False
            
            if len(str(df['Trimming Dimension for Resize'][i])) > 3:
                    insta_story = True
            while True:
                ret, frame = cap.read()
                if ret == True:
                    b = cv2.resize(crop_image(frame, insta_story, str(df['Trimming Dimension for Resize'][i])), (1280,720), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
                    out.write(b)
                else:
                    break
            out.release()
            cap.release()
            cv2.destroyAllWindows()
            
