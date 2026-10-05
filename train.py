from multiprocessing import freeze_support

from classify import Classifier


def main() -> None:
    clf = Classifier(cfg="cfg/car_quality.yaml")
    ckpt = (r"output/20260616/CarQuality.Classification/mobilenet_v3_small"
            r"/version_11/checkpoints/best_loss_epoch=36-val_loss=0.066-val_acc=0.985.ckpt")

    clf.train(
        data_dir="datasets/dataset_4",
        ckpt_path=None,
        num_classes=2,
        precision="32",
        resize=256,
        enable_mlflow=False,
        use_class_weights=False,
        optimizer="MuSGD",
        learning_rate=0.001,
        scheduler="Cosine",
        max_epochs=100,
    )


if __name__ == "__main__":
    main()
