import cv2
"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import random
import time

from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet


def run(model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(1)
    print("is camera open", cap.isOpened())
    cap.set(3,320)
    cap.set(4,240)
    print("start processing")

    i = 0
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        print("new frame", ret)
        p1 = time.time()
        print(f"take a picture {p1 - start}")
        if ret:
            img = utils.process_camera_img(frame)
            img_input = transform({"image": img})["image"]
            p2 = time.time()
            print(f"transoform image {p2 - p1}")
            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                p3 = time.time()
                print(f"from numpy to cuda {p3 - p2}")
                prediction = model.forward(sample)
                p4 = time.time()
                print(f"prediction {p4 - p3}")
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                p5 = time.time()
                print(f"prediction from cuda to cpu {p5 - p4}")


            # output

            r = random.randint(0, 10000)
            cv2.imwrite(f"output/input-{i}-{r}.png", frame)
            utils.write_depth(f"output/depth-{i}-{r}", prediction, bits=2)
            p6 = time.time()
            print(f"save input and write depth {p6 - p5}")

            cv2.imshow('frame', frame)
            cv2.imshow('prediction', prediction)
            p7 = time.time()
            print(f"show images {p7 - p6}")
            i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Camera is not recording")
        print(f"image took {time.time() - start} s")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("finished")


if __name__ == "__main__":
    MODEL_PATH = "model.pt"

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(MODEL_PATH)
