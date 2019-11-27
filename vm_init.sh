sudo apt install htop
sudo pip3 install gast==0.2.2

gcloud config set compute/region europe-west4
gcloud config set compute/zone europe-west4-a
# Not adding this as this should be done on a tmux basis
# export TPU_NAME=tpu-test
# BERT
git clone https://github.com/DavidNemeskey/bert.git
# ALBERT in the albert directory
git clone https://github.com/DavidNemeskey/google-research.git
