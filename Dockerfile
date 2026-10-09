FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-devel

RUN pip install matplotlib numpy pandas scipy scikit-image scikit-learn tifffile tqdm torchvision

ADD evaluate.py /evaluate.py