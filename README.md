# Pi inferences

This repository contains scripts to run inferences on embedded device. The scripts firstly perform insect detection on images, followed by species classification of the insect crops. 


## Set-up on Pi

### Install Python3.9

```
sudo apt-get update
sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev
````

```
wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
tar xf Python-3.9.0.tar.xz
cd Python-3.9.0
./configure --enable-optimizations --prefix=/usr
make
```

```
sudo make altinstall
```

```
cd ..
sudo rm -r Python-3.9.0
rm Python-3.9.0.tar.xz
. ~/.bashrc
```

### Create venv

```
python3.9 -m venv ./venv_3.9
source venv_3.9/bin/activate
```

Update pip

```
/home/pi/Desktop/model_data_bookworm/venv_3.9/bin/python3.9 -m pip install --upgrade pip
```

```
pip install -r requirements.txt
```
