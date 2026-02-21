!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="c6ODS5WM37hWdEQzcoSJ")
project = rf.workspace("helmetdetection-bvukf").project("helmet-detection-rszll")
version = project.version(3)
dataset = version.download("yolov8")
                