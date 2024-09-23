import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from scipy import linalg
import numpy as np
from PIL import Image
import time
import psutil
from pthflops import count_ops


def get_activations(images, model, batch_size=32, dims=2048):
    model.eval()
    n_batches = len(images) // batch_size + 1
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = torch.stack([transforms.ToTensor()(img) for img in images[start:end]])
        batch = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(batch)
        batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)[0]
            ops = count_ops(model, batch)
        pred_arr[start:end] = pred.cpu().numpy().reshape(batch.shape[0], -1)

    return pred_arr[:len(images)]

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(images1, images2):
    start_time = time.time()
    start_cpu_percent = psutil.cpu_percent()
    start_memory = psutil.virtual_memory().used

    model = inception_v3(pretrained=True, transform_input=False).cuda()
    model.fc = torch.nn.Identity()

    act1 = get_activations(images1, model)
    act2 = get_activations(images2, model)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    end_time = time.time()
    end_cpu_percent = psutil.cpu_percent()
    end_memory = psutil.virtual_memory().used

    time_taken = end_time - start_time
    cpu_usage = (end_cpu_percent + start_cpu_percent) / 2
    memory_usage = end_memory - start_memory

    return fid_value, time_taken, cpu_usage, memory_usage

def run_multiple_times(images1, images2, num_runs=10000):
    total_fid = 0
    total_time = 0
    total_cpu = 0
    total_memory = 0

    for i in range(num_runs):
        print(f"실행 {i+1}/{num_runs}")
        fid, time_taken, cpu_usage, memory_usage = calculate_fid(images1, images2)
        total_fid += fid
        total_time += time_taken
        total_cpu += cpu_usage
        total_memory += memory_usage

    avg_fid = total_fid / num_runs
    avg_time = total_time / num_runs
    avg_cpu = total_cpu / num_runs
    avg_memory = total_memory / num_runs

    return avg_fid, avg_time, avg_cpu, avg_memory

# 이미지 로드 (예시)
image1 = Image.open("../Research/data/detect/coco/train2017/000000159203.jpg")
image2 = Image.open("../Research/data/detect/coco/train2017/000000159262.jpg")
num_runs = 100

avg_fid, avg_time, avg_cpu, avg_memory = run_multiple_times([image1], [image2], num_runs)

print(f"평균 FID 스코어: {avg_fid:.2f}")
print(f"평균 소요 시간: {avg_time:.2f}초")
print(f"평균 CPU 사용률: {avg_cpu:.2f}%")
print(f"평균 메모리 사용량: {avg_memory / (1024 * 1024):.2f} MB")

