from ultralytics import YOLO
from PIL import Image

objectName = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


objectCategory = {
    0: "Special Handling",
    1: "Recyclable",
    2: "Recyclable",
    3: "Recyclable",
    4: "Recyclable",
    5: "Recyclable",
    6: "Recyclable",
    7: "Recyclable",
    8: "Recyclable",
    9: "Trash",
    10: "Trash",
    11: "Trash",
    12: "Trash",
    13: "Trash",
    14: "Compost",
    15: "Compost",
    16: "Compost",
    17: "Compost",
    18: "Compost",
    19: "Compost",
    20: "Compost",
    21: "Compost",
    22: "Compost",
    23: "Compost",
    24: "Recyclable",
    25: "Recyclable",
    26: "Recyclable",
    27: "Recyclable",
    28: "Recyclable",
    29: "Recyclable",
    30: "Recyclable",
    31: "Recyclable",
    32: "Recyclable",
    33: "Recyclable",
    34: "Recyclable",
    35: "Recyclable",
    36: "Recyclable",
    37: "Recyclable",
    38: "Recyclable",
    39: "Recyclable",
    40: "Recyclable",
    41: "Recyclable",
    42: "Recyclable",
    43: "Recyclable",
    44: "Recyclable",
    45: "Recyclable",
    46: "Compost",
    47: "Compost",
    48: "Compost",
    49: "Compost",
    50: "Compost",
    51: "Compost",
    52: "Compost",
    53: "Compost",
    54: "Compost",
    55: "Compost",
    56: "Recyclable",
    57: "Recyclable",
    58: "Recyclable",
    59: "Recyclable",
    60: "Recyclable",
    61: "Recyclable",
    62: "Recyclable",
    63: "Recyclable",
    64: "Recyclable",
    65: "Recyclable",
    66: "Recyclable",
    67: "Recyclable",
    68: "Recyclable",
    69: "Recyclable",
    70: "Recyclable",
    71: "Recyclable",
    72: "Recyclable",
    73: "Recyclable",
    74: "Recyclable",
    75: "Recyclable",
    76: "Recyclable",
    77: "Trash",
    78: "Recyclable",
    79: "Recyclable",
}


class CVObject:
    def __init__(self, id):
        self.name = objectName[id]
        self.category = objectCategory[id]


class ModelTrainer:
    def __init__(self, path="yolov8n.yaml"):
        self.model = YOLO(path)  # build a new model from scratch

    def train(self, path, epoch=3):
        self.model.train(
            data=path,
            epochs=epoch,
        )  # train the model

    def export(self):
        self.metrics = self.model.val()
        self.path = self.model.export(format="onnx")


class Classifier:
    def __init__(self, path):
        self.model = YOLO(path)

    def processImage(self, path):
        self.result = self.model.predict(path)[0]

    def createObjects(self):
        self.objects = []
        for box in self.result.boxes:
            id = int(box.cls[0].item())
            self.objects.append(CVObject(id))

    def displayImage(self):
        Image.fromarray(self.result.plot()[:, :, ::-1]).show()
