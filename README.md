# A (Heavily Documented) TensorFlow Implementation of Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow == 1.2
  * librosa
  * tqdm

## Data
Since the [original paper](https://arxiv.org/abs/1703.10135) was based on their internal data, I use a freely available one, instead.

[The World English Bible](https://en.wikipedia.org/wiki/World_English_Bible) is a public domain update of the American Standard Version of 1901 into modern English. Its text and audio recordings are freely available [here](http://www.audiotreasure.com/webindex.htm). Unfortunately, however, each of the audio files matches a chapter, not a verse, so is too long for many machine learning tasks. I had someone slice them by verse manually. You can download [the audio data](https://dl.dropboxusercontent.com/u/42868014/WEB.zip) and its [text](https://dl.dropboxusercontent.com/u/42868014/text.csv) from my dropbox.



## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `prepro.py` loads vocabulary, training/evaluation data.
  * `data_load.py` loads data and put them in queues so multiple mini-bach data are generated in parallel.
  * `utils.py` has several custom operational functions.
  * `modules.py` contains building blocks for encoding/decoding networks.
  * `networks.py` has three core networks, that is, encoding, decoding, and postprocessing network.
  * `train.py` is for training.
  * `eval.py` is for sample synthesis.
  

## Training
  * STEP 1. Adjust hyper parameters in `hyperparams.py` if necessary.
  * STEP 2. Download and extract [the audio data](https://dl.dropboxusercontent.com/u/42868014/WEB.zip) and its [text](https://dl.dropboxusercontent.com/u/42868014/text.csv).
  * STEP 3. Run `train.py`. or `train_multi_gpus.py` if you have more than one gpu.

## Sample Synthesis
  * Run `eval.py` to get samples.

### Acknowledgements
I would like to show my respect to Dave, the host of www.audiotreasure.com and the reader of the audio files.
