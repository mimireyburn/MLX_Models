wget -P ~/ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/Miniconda3-latest-Linux-x86_64.sh
~/Miniconda3-latest-Linux-x86_64.sh -b
export PATH=~/miniconda3/bin:$PATH
conda init
# close and start a new session
conda config --set auto_activate_base fals
# conda environment
conda create --name foo-bar-env python=3.8 -y
conda env list
conda activate foo-bar-env
conda list
# utils & server
pip install pipx
pipx run nvito
print "finished"
echo "finished"