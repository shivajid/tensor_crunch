sudo apt update
sudo apt install python3.11
sudo apt install python3-venv
#Create venv with python3.11

python3.11 -m venv venv

source venv/bin/activate


# Install tunix
git clone https://github.com/google/tunix.git

#Change directory of tunix
cd tunix

#Install tunix
pip install .

#Clone Tensor Crunch
https://github.com/shivajid/tensor_crunch.git

#Install Qwix
pip install git+https://github.com/google/qwix

cd tensor_crunch

#Install the required packages
pip install -r requirements.txt



