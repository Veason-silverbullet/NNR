# the versions of torch and torchtext must be matched (https://pypi.org/project/torchtext)
# the CUDA version must be matched with torch-scatter (https://github.com/rusty1s/pytorch_scatter)
TORCH_VERSION=1.12.1
TORCHTEXT_VERSION=0.13.1
CUDA_VERSION=cu113


pip install torch==${TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip install torchtext==${TORCHTEXT_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip install nltk
pip install numpy
pip install sklearn
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
