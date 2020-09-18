import os
import glob
import argparse
import numpy as np

class LabelToKittiConverter:
    """ KITTI FORMAT
    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    """
    def __init__(self, conversion, labels_fp, gt_labels_fp, csv):
        conversion_dict = self.get_attribute_idx(conversion)
        self.csv = csv
        if labels_fp is not None:
            print("------ Prediction Labels Conversion -------")
            new_labels_fp = (os.path.join(labels_fp, '..',  'pred_labels_kitti/data/'))
            if new_labels_fp[0] == '/':
                new_labels_fp = new_labels_fp[1:]
            os.makedirs(new_labels_fp, exist_ok=True)
            self.convert_to_kitti(conversion_dict, labels_fp, new_labels_fp, pred_labels=True)
        if gt_labels_fp is not None:
            print("------ Ground Truth Labels Conversion -------")
            new_labels_fp = os.path.join(gt_labels_fp, '..',  '/gt_labels_kitti/')
            if new_labels_fp[0] == '/':
                new_labels_fp = new_labels_fp[1:]
            os.makedirs(new_labels_fp, exist_ok=True)
            self.convert_to_kitti(conversion_dict, gt_labels_fp, new_labels_fp, pred_labels=False)

    def convert_to_kitti(self, conversion_key, fp, new_fp, pred_labels=False):
        labels_list = glob.glob(fp + "*")
        for num, example in enumerate(labels_list):
            print("{}/{}\r".format(num+1, len(labels_list)))
            example_idx = os.path.basename(example)
            new_label = self.new_label_from_txt(example, conversion_key, pred=pred_labels)
            if pred_labels:
                np.savetxt(new_fp + example_idx, new_label, delimiter=' ',
                           fmt='%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s')
            else:
                np.savetxt(new_fp + example_idx, new_label, delimiter=' ',
                           fmt='%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s')

    def new_label_from_txt(self, label_path, idx_key, pred=True):
        classes = []
        trunc = []
        occ = []
        obs = []
        camera_box = []
        score = []
        x, y, z, r = [], [], [], []
        l, w, h = [], [], []
        with open(label_path, "r") as f:
            labels = f.read().split("\n")
            for label in labels:
                if not label:
                    continue
                if self.csv:
                    label = label.replace(" ", "")
                    label = label.split(",")
                else:
                    label = label.replace(",", "")
                    label = label.split(" ")

                if 'class' in idx_key:
                    classes.append(label[idx_key['class']])
                else:
                    classes.append(['Car']) # assume if no class is specified, its a car
                if 'truncated' in idx_key:
                    trunc.append(label[idx_key['truncated']])
                else:
                    trunc.append(0.00)
                if 'occluded' in idx_key:
                    occ = occ.append(label[idx_key['occluded']])
                else:
                    occ.append(0)
                if 'alpha' in idx_key:
                    obs.append(label[idx_key['alpha']])
                else:
                    obs.append(0)
                if 'x1' in idx_key:
                    start, end = idx_key['x1'], idx_key['x1']+4
                    camera_box.append(label[start:end])
                else:
                    camera_box.append((0, 0, 50, 50))
                if 'x' in idx_key:
                    x.append(label[idx_key['x']])
                else:
                    x.append(0)
                if 'y' in idx_key:
                    y.append(label[idx_key['y']])
                else:
                    y.append(0)
                if 'z' in idx_key:
                    z.append(label[idx_key['z']])
                else:
                    z.append(0)
                if 'r' in idx_key:
                    r.append(label[idx_key['r']])
                else:
                    r.append(0)
                if 'l' in idx_key:
                    l.append(label[idx_key['l']])
                else:
                    l.append(0)
                if 'w' in idx_key:
                    w.append(label[idx_key['w']])
                else:
                    w.append(0)
                if 'h' in idx_key:
                    h.append(label[idx_key['h']])
                else:
                    h.append(0)
                if pred:
                    if 'score' in idx_key:
                        score.append(label[idx_key['score']])
        final_array = np.hstack((
            np.array(classes).reshape(-1, 1),
            np.array(trunc).reshape(-1, 1),
            np.array(occ).reshape(-1,1 ),
            np.array(obs).reshape(-1, 1),
            np.array(camera_box),
            np.array(h).reshape(-1, 1),
            np.array(w).reshape(-1, 1),
            np.array(l).reshape(-1, 1),
            np.array(x).reshape(-1, 1),
            np.array(y).reshape(-1, 1),
            np.array(z).reshape(-1, 1),
            np.array(r).reshape(-1, 1)
        ))
        if pred:
            final_array = np.hstack((final_array, np.array(score).reshape(-1, 1)))
        return final_array

    @staticmethod
    def get_attribute_idx(conversion):
        idx_dict = {}
        conversion = conversion.split(" ")
        print("-- Your conversion key --")
        for i, attribute in enumerate(conversion):
            print(i, attribute)
            if attribute == 'class':
                idx_dict['class'] = i
            elif attribute == 'truncated':
                idx_dict['truncated'] = i
            elif attribute == 'occluded':
                idx_dict['occluded'] = i
            elif attribute == 'alpha':
                idx_dict['alpha'] = i
            elif attribute == 'x1':
                idx_dict['x1'] = i # assumes [x1 y1 x2 y2] follows
            elif attribute == 'x':
                idx_dict['x'] = i
            elif attribute == 'y':
                idx_dict['y'] = i
            elif attribute == 'z':
                idx_dict['z'] = i
            elif attribute == 'l':
                idx_dict['l'] = i
            elif attribute == 'w':
                idx_dict['w'] = i
            elif attribute == 'h':
                idx_dict['h'] = i
            elif attribute == 'r':
                idx_dict['r'] = i
            elif attribute == 'score':
                idx_dict['score'] = i
        return idx_dict


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_labels', type=str, help='file path with your prediction labels')
    parser.add_argument('--gt_labels', type=str, help='file path with your ground truth labels')
    parser.add_argument('--format', type=str, help='your label format e.g. "class, x, y, z, l, w, h"', required=True)
    parser.add_argument('--csv', dest='csv', action='store_true', help='is your file csv or space separated')
    args = parser.parse_args()
    print("------ Converting to KITTI Labels -------")
    LabelToKittiConverter(args.format, args.pred_labels, args.gt_labels, csv=args.csv)


if __name__ == '__main__':
    main()
