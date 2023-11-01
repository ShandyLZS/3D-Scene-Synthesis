import os 
import json

def read_file(path, result):
    files = os.listdir(path)
    for idx, file in enumerate(files):
        # 0a761819-05d1-4647-889b-a726747201b1_MasterBedroom-24539_84.hdf5
        scene_json, room_id, _ = file.split('_')
        if [scene_json, room_id] in result:
            continue
        else:
            result.append([scene_json, room_id])
    print(idx)
    return result

if __name__ == '__main__':
    path = 'datasets/3D-Front/3D-FRONT_samples/bed'
    result = []
    file_info = read_file(path, result)
    with open('./utils/threed_front/vis/gt_sample_large.json', 'w') as f:
        json.dump(file_info, f)