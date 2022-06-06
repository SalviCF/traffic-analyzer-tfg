### Description
Proccess images with *YOLOv4* and saves them into a *MinIO* server.

### Instalation
Clone this repository:
```shell
git clone https://github.com/SalviCF/traffic-analyzer-tfg.git
```

### Settings
Setup your environmental variables by creating a `.env` file taking `.env.template` has example (using your own connection settings).
Download the [configuration files for *YOLOv4*](https://drive.google.com/drive/folders/1ZLlNqTUirG8Kfm-LftFJ67dkCz4xFa28).

### Usage
Open a terminal inside your local repository an run:
```shell
python run_detectors.py
```
