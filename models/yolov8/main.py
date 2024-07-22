from ultralytics import YOLO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    DS_PATH = 'dataset/deeprad.yaml'

    parameters = {
        'data': DS_PATH,
        'epochs': 30,
        'batch': 5,
        'lr0': 1e-4,
        'lrf': 1e-4,
        'optimizer': 'Adam',
    }

    #model = YOLO('yolov8n-seg.yaml', task='segment') # creating from scratch
    model = YOLO('yolov8n-seg.pt')   # loading pretrained

    results = model.train(**parameters)

    model.val()

    pred = model('/content/datasets/data/val/imagem-003.jpg')

    for p in pred:
        im = p.plot()
        plt.imshow(im) # se nao tiver nenhuma mascara provavel que nao treinou o suficiente