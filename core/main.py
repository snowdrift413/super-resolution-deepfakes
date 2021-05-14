import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import glob
from skimage.transform import resize
from skimage import img_as_ubyte
from IPython.display import HTML
from .face_alignment.api import FaceAlignment
from .face_alignment.api import LandmarksType
import warnings
import datetime
import math
import os, sys
import shutil
import yaml
from argparse import ArgumentParser
import subprocess
from tqdm import tqdm
import tensorflow as tf
import torch
from .first_order_model.sync_batchnorm import DataParallelWithCallback
from .first_order_model.modules.generator import OcclusionAwareGenerator
from .first_order_model.modules.keypoint_detector import KPDetector
from .first_order_model.animate import normalize_kp
from scipy.spatial import ConvexHull
warnings.filterwarnings("ignore")

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def preprocess(source_image, driving_video):
    source = imageio.imread("source/" + str(source_image))
    image_type = ""
    for i in source_image[::-1]:
        if i != ".":
            image_type+=i
        else:
            break
    image_type = image_type[::-1]
    driving = imageio.mimread("driving/" + str(driving_video), memtest=False)
    source = resize(source, (256, 256))[..., :3]
    driving = [resize(frame, (256, 256))[..., :3] for frame in driving]
    reader = imageio.get_reader("driving/" + str(driving_video), "ffmpeg")
    fps = int(math.floor(reader.get_meta_data()["fps"]))
    return source, driving, fps, image_type

def display(source=None, driving=None, generated=None):
    number_of_videos = 0
    length = 0
    if source is not None:
        number_of_videos += 1
    if driving is not None:
        number_of_videos += 1
        length = len(driving)
    if generated is not None:
        number_of_videos += 1
        length = len(generated)
    fig = plt.figure(figsize=(5 * number_of_videos, 5))

    ims = []
    for i in range(length):
        cols = []
        if source is not None:
            cols.append(source)
        if driving is not None:
            cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return HTML(ani.to_html5_video())
    
def generate(source_image, driving_video, image_type, fps, best_frame=False, relative=True, adapt_movement_scale=True, improve=False, config="vox-256.yaml", checkpoint="vox-cpk.pth.tar", cpu=True):
    generator, kp_detector = load_checkpoints(config_path="core/first_order_model/config/" + str(config), 
                            checkpoint_path="core/first_order_model/checkpoints/" + str(checkpoint), cpu=cpu)

    print("Making animations")
    if best_frame == True:
        best_frame_i = find_best_frame(source=source_image, driving=driving_video, cpu=cpu)
        driving_forward = driving_video[best_frame_i:]
        driving_backward = driving_video[:(best_frame_i+1)][::-1]
        predictions_forward = make_animation(source_image=source_image, driving_video=driving_forward, generator=generator, kp_detector=kp_detector, relative=relative, adapt_movement_scale=adapt_movement_scale, cpu=cpu)
        predictions_backward = make_animation(source_image=source_image, driving_video=driving_backward, generator=generator, kp_detector=kp_detector, relative=relative, adapt_movement_scale=adapt_movement_scale, cpu=cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image=source_image, driving_video=driving_video, generator=generator, kp_detector=kp_detector, relative=relative, adapt_movement_scale=adapt_movement_scale, cpu=cpu)
    time = datetime.datetime.now().strftime("%d%b%Y%H%M")
    video = f"generated_imageio{time}.mp4"
    imageio.mimsave(f"videos/{video}", [img_as_ubyte(image) for image in predictions], fps = fps)
    if improve == True:
        print("Improving quality")
        improve_quality(images=predictions, image_type=image_type, fps=fps, cpu=cpu)
    return predictions

def improve_quality(images, image_type, fps, cpu):
    shutil.rmtree("tmp/") 
    os.mkdir("tmp/")
    count = 1
    for image in images:
        imageio.imwrite(f"tmp/{count:012d}.{image_type}", img_as_ubyte(image))
        count += 1

    if not cpu:
        tf.device("/device:GPU:0")
    else:
        tf.device("/device:CPU:0")

    images = [cv2.imread(image, 1) for image in glob.glob(f"tmp/*.{image_type}")]
    model = tf.keras.models.load_model("core/fast_srgan/models/generator.h5", compile=False)
    inputs = tf.keras.Input((images[0].shape[0], images[0].shape[1], 3))
    output = model(inputs)
    model = tf.keras.models.Model(inputs, output)
    count = 1
    for image in tqdm(images):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = model.predict(np.expand_dims(image, axis=0))[0]
        image = ((image + 1) / 2.) * 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"tmp/{count:012d}.{image_type}", image)
        count += 1
    
    time = datetime.datetime.now().strftime("%d%b%Y%H%M")
    video = f"generatedimp_cv2{time}.mp4"
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(f"videos/{video}", 0, fourcc, fps, (images[0].shape[0], images[0].shape[1]))
    for i in range(len(images)):
        #images_improved[i] = cv2.normalize(images_improved[i], None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        out.write(images[i])
    out.release()
    try:
        subprocess.call(["ffmpeg", "-r", "30", "-pattern_type", "glob", "-i", f"tmp/*.{image_type}", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "0", "-r", "30", f"videos/generatedimp_ff{time}.mp4"])
    except Exception as e:
        print(e)

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = FaceAlignment(LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num