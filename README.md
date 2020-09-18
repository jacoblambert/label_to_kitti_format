If you have some labels you want to evaluate on KITTI benchmark but have format problems, here some quick scripts for you to convert to KITTI format and evaluate using their metrics.

#### KITTI Converter
You can convert your labels of arbitrary format to KITTI labels easily using this script. Example:
```
python to_kitti_converter.py --pred_labels path_to/pred_dir --gt_labels path_to/pred_dir --format "class x y z l w h r score
```

| Parameter | Flag | Type | Default | Description |
|-----|--------|------|-----|-----|
| Format | --format | list | None, *required* | Label format, see below |
| Gt label path | --gt_labels | str | None | Groundtruth label file path | 
| Pred label path | --pred_labels | str | None | Prediction label file path, same format as GT but with scores and stored in data/ folder | 
| CSV file | --csv | flag | False | If your file is a CSV, use this flag. Otherwise assume space seperated like KITTI)

You define your label format based on which KITTI fields you have. KITTI labels use the following field and identifier:

| KITTI Field | Converter ID | Notes
|-------|-------|-------|
| Type | class | |
| Truncated | truncated | |
| Occluded | occluded |  |
| Alpha | alpha |  |
| 2D bbox [x1 y1 x2 y2]| x1 y1 x2 y2 | Must be side by side in this order |
| 3D shape [h, w, l] | h w l | Any order is fine |
| 3D bbox [x, y, z]| x y z | Any order is fine |
| Rotation angle | r | radians |
| Score | score | Set to 1 if none is given for prediction labels |

For example, say your label has only has classes, 3D Bbox and 3D Bbox sizes and scores such as like:
```
Car 10 -30 1.2 3.6 1.9 2.01 0.9
```
Then you would use:
```
--format "class x y z l w h score"
```
Say your format only has BEV bbox (lidar frame) and has extra fields like:
```
Timestamp Class Color BEV_x BEV_y Length Width Score
```
Then your format would be:
```
--format "timestamp class color x y l w score"
```
and the timestamp and color fields would not be used. Say you want to evaluate on image results, then your format might be like
```
--format "class x1 y2 x2 y2 score"
```
Extra fields need an ID, but anything outside the above list is fine. You don't need to worry too much about frame of reference, as long as predictions and groundtruth match the KITTI c++ code will work.

#### Official "offline" KITTI Eval
Make sure you have prediction and grouthtruth folders in KITTI format. You also this directory format
```
result_dir
├── data
│   ├── 000000.txt ... 00xxxx.txt
gt_dir
├── 000000.txt ... 00xxxx.txt
```
Then in kitti_cpp_eval/ run:
```
./evaluate_object_3d_offline gt_dir result_dir
```

To recompile it just run in the cpp folder:
```
g++ -O3 -DNDEBUG -o evaluate_object_offline evaluate_object_offline.cpp
```
