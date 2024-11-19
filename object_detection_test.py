import torch
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI
from typing import Dict
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve import Application
from pydantic import BaseModel

app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle: DeploymentHandle):
        self.handle = object_detection_handle

    @app.get(
        "/detect",
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def detect(self, image_url: str):
        image = await self.handle.detect.remote(image_url)
        file_stream = BytesIO()
        image.save(file_stream, "jpeg")
        return Response(content=file_stream.getvalue(), media_type="image/jpeg")

    def reconfigure(self, config: Dict):
        self._threshold = config["threshold"]
        print("Threshold:", self._threshold)

@serve.deployment(
    # ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class ObjectDetection:
    def __init__(self, message: str, code: int):
        self._message = message
        self._code = code
        print("Message:", self._message)
        print("Code:", self._code)
        ROOT = 'D:/alg/ultralytics_yolov5_master'
        # self.model = torch.hub.load(ROOT, 'custom', source = 'local', path = ROOT +  '/yolov5s.pt')
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        # self.model.cuda()
        # self.model.to(torch.device(0))

    def detect(self, image_url: str):
        result_im = self.model(image_url)
        return Image.fromarray(result_im.render()[0].astype(np.uint8))

    def reconfigure(self, config: Dict):
        self._language = config["language"]
        print("Language:", self._language)
        self._type = config["type"]
        print("Type:", self._type)


def app_builder(args: Dict[str, str]) -> Application:
    return APIIngress.bind(ObjectDetection.bind(args["message"], args["code"]))

# class HelloWorldArgs(BaseModel):
#     message: str
#
# def typed_app_builder(args: HelloWorldArgs) -> Application:
#     return ObjectDetection.bind(args.message)


# entrypoint = APIIngress.bind(ObjectDetection.bind())

