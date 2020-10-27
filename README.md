# MediaPipe Hand Landmark Model

An unofficial Implementation of Hand Landmark Model using Tensorflow 2.0.

## Usage

### Install dependencies

```
pip install -r requirements.txt
```

### Training

**Prepare training and validation data**

- Directory Structure

    In directory `data`, create `image` and `annotation` folders

    ```
    dataset
        |----train_image
        |----train_annotation
        |----val_image
        |----val_annotation
    ```
    
- Raw Annotation Format

    Each image need a corresponding Json file that contains all hand items in the image.

    Each item in a json file should have 3 attributes `label` and `landmark`
    
    `label`: always be `1` (only one class)
        
    `landmark`: 21 key points in total, [x1, y1, x2, y2,... x21, y21]
    
    ```
    [
      {
        "label": "1",
        "landmark": [
          [128.54510, 123.11315],
          [253.02255, 53.02255]
          ...
        ],
      }
    ]
    ```
    
- Use the conversion code in `development.ipynb` to convert (crop) raw data to training data

    ```python
    for image_path in all_image_path:
    file_name = os.path.split(image_path)[-1].split('.')[0]
    raw_image = cv2.imread(os.path.join('data', 'raw', 'image', file_name + '.jpg'))
    with open(os.path.join('data', 'raw', 'annotation', file_name + '.json')) as json_file: anno = json.load(json_file)

    for i, landmarks in enumerate(anno):
        landmarks = np.array(landmarks["landmark"])

        w = max_distance((landmarks)[[0, 1, 2, 5, 9, 13, 17]])
        triangle = get_triangle(landmarks[0], landmarks[9], w)
        matrix = cv2.getAffineTransform(triangle, TARGET_TRIANGLE)

        input_image = cv2.warpAffine(raw_image, matrix, (256, 256))
        encoded_landmarks = encode_landmarks(landmarks, matrix)
        
        cv2.imwrite(os.path.join('data', 'image', file_name + str(i) + '.jpg'), input_image)
        
        output = []
        item = Object()
        item.landmark = encoded_landmarks.tolist()
        with open(os.path.join('data', 'annotation', file_name  + str(i) + '.json'), 'w') as f_out: json.dump(item.__dict__, f_out)
    ```

**Run `train.py`**

- Check the training config in `train.py` is correct then run.

### Inference

TODO

## Result

TODO

## TODO

- Hand presence and handedness output

- Train a model on open dataset

- Inference and visualize

## Reference

- [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10214)
- [ssd_keras](https://github.com/pierluigiferrari/ssd_keras)
- [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
