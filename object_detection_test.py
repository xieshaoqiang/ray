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


@serve.deployment(
    # ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class ObjectDetection:
    def __init__(self):
        # self._message = message
        # print("Message:", self._message)
        ROOT = 'D:/alg/ultralytics_yolov5_master'
        # self.model = torch.hub.load(ROOT, 'custom', source = 'local', path = ROOT +  '/yolov5s.pt')
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        # self.model.cuda()
        # self.model.to(torch.device(0))

    def detect(self, image_url: str):
        result_im = self.model(image_url)
        return Image.fromarray(result_im.render()[0].astype(np.uint8))

    def reconfigure(self, config: Dict):
        self.threshold = config["threshold"]


# def app_builder(args: Dict[str, str]) -> Application:
#     return ObjectDetection.bind(args["message"])

# class HelloWorldArgs(BaseModel):
#     message: str
#
# def typed_app_builder(args: HelloWorldArgs) -> Application:
#     return ObjectDetection.bind(args.message)


entrypoint = APIIngress.bind(ObjectDetection.bind())
