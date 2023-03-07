from tqdm import tqdm
import cv2
import os

size = (640, 480)

src_path = './video_data/room2/images_split'
sav_path = './render.mp4'

all_files = sorted(os.listdir(src_path))
index = len(all_files)
print("total number: ", index)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowrite = cv2.VideoWriter(sav_path, fourcc, 6, size)
img_array=[]
 
for fn in tqdm(all_files):
    file_path = os.path.join(src_path, fn)
    img = cv2.imread(file_path)
    if img is None:
        print(file_path + " is error!")
        continue
    img_array.append(img)

for i in tqdm(range(0,index)):
    img_array[i] = cv2.resize(img_array[i], size)
    videowrite.write(img_array[i])

print('------done!!!-------')