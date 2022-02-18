# Scripts to benchmark the Essentia Models


## Requirements
Install the Python requirements of this repository from `requirements.txt`.

To use a GPU, both the CUDA and CuDNN libraries are needed.
The last essnetia-tensorflow wheels were built with `CUDA==11.2` and `CuDNN==8.1`.
However minor version variations may work.
For example, the libraries were correctly detected in a Conda environment with `Python 3.9`, `cudatoolkit==11.3.1` and `cudnn==8.2.1`.

## Configuration
The benchmark parameters are in `cfg/config.json`:
  - **models_base**: path to the base folder of the models. The models should follow the same hierarchy as in the [web repository](essentia.upf.edu/models/).
  - **benchmark_name**: this name will appear in the results json file.
  - **audio_duration**: seconds of audio that are tested. This bechmark does not load any real audio, it is emulated via `numpy.ones()`.
  - **repetitions**: number of repetitions to perform. According to [timeit](https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat), the reported time is the minimum of all the repetitions.
  - **platform**: rather `gpu` or `cpu`. Note that Essentia has no way to verify that a GPU exists and is available, or that the required versions of CUDA and CuDNN are installed. If any of these conditions fails, inference will happen on the cpu. It is the user's responsibility to check the TF logs to verify the the inference happens in the GPU when it is intended. 
  - **gpu_id**: gpu id to use.

The `cfg/routines.json` file describes the models to test.

## Usage
Run `./src/benchmark.py`.



