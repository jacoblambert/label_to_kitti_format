import json
import numpy as np

label_file = "json_labels.json"
save_folder = "/home/jacob/Documents/data/deepen_data/gt_labels/"
with open(label_file) as f:
    loaded_json = json.load(f)
for pcd in loaded_json['labels'].keys():
    id = pcd[:-4]
    pcd_dict = loaded_json['labels'][pcd]
    pcd_labels = np.zeros((len(pcd_dict), 7))
    label_id = []
    for i, label in enumerate(pcd_dict):
        if label['label_category_id'] == 'car' or label['label_category_id'] == 'kei':
            label_id.append('Car')
            print(label['label_category_id'])
        x = label['three_d_bbox']['cx']
        y = label['three_d_bbox']['cy']
        z = label['three_d_bbox']['cz']
        l = label['three_d_bbox']['l']
        w = label['three_d_bbox']['w']
        z = label['three_d_bbox']['h']
        r = label['three_d_bbox']['rot_z']
        pcd_labels[i, :] = np.array([x, y, z, l, w, z, r])
    label_ids = np.array(label_id).reshape(-1, 1)
    pcd_labels = np.hstack((label_ids, pcd_labels))
    np.savetxt(save_folder + id + '.txt', pcd_labels, delimiter=' ',
               fmt='%s %s %s %s %s %s %s %s')