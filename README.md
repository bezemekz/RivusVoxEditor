# RivusVoxEditor
RivusVox Editor: the world's first near-live zero-shot adaptive speech editing system

![](https://github.com/user-attachments/assets/c913fa06-2974-4535-9dba-e31b20141155)

By Zack Bezemek and Francesca Balestrieri.

## 0. Project description

The main goal behind our project was to be able to edit speech on the fly, by removing key words and replacing them with other words with a generated voice as similar to the speaker's voice as possible, all while having no prior knowledge about the speaker. A possible concrete application we had in mind is live streaming: streamers usually stream with some delay, and they might find useful to exploit that delay to automatically edit any slip-ups on the go, or edit key words for entertainment purposes.  
We had the following two main objectives for our project:

(1) to improve the inference time and the speech editing mechanics of one the current state-of-the-art speech editing models ([FluentSpeech](https://github.com/Zain-Jiang/Speech-Editing-Toolkit));

(2) using our improved model, to be able to edit speech during a live streaming with a maximum fixed delay of possibly 10s.

Our model currently only works for speech in English (and, in particular, the best results are obtained for speech in American English; this is due to the fact that the training corpus consisted mainly of American English speakers). While the model is *zero-shot*, meaning that it doesn't require any additional training or fine-tuning for priorly unseen speakers, we have also experimented with fine-tuning certain components of the model to an individual speaker (using roughly 4 hours of clean speech data from that speaker) with a British accent - thus with an accent quite distant from that of the training corpus. (For the results of these fine-tuning experiments, please see the section **Fine-tuning on individual speakers** below.) Our model is also *near-live*, meaning that the inference is done almost in real-time (usually in less than a second). Exploiting this feature, in line with our objectives we developed a demo of speech editing while live-streaming audio from different sources (e.g. the user's microphone, a .wav audio file in the user's local machine, or a youtube link); for more information, please see the section **Implementing a live streaming speech editing demo** below.

##  1. Improving the inference time of the base model, and fine-tuning on single speaker for enhanced performance.

#### **An overview of the modifications/improvements over the base model**

We took as our base model the (non-stutter-prediction components of the) *FluentSpeech* model developed by Ziyue Jiang, Qian Yang, Jialong Zuo, Zhenhui Ye, Rongjie Huang, Yi Ren, and Zhou Zhao, which is a automatic speech editing architecture using a context-aware diffusion models to iteratively refine the edited mel-spectrograms conditional on context features. For a thorough description of how *FluentSpeech* works, we refer to the corresponding [arXiv paper](https://arxiv.org/abs/2305.13612) and [github repository](https://github.com/Zain-Jiang/Speech-Editing-Toolkit). The architecture of *FluentSpeech* is the following (see [https://arxiv.org/abs/2305.13612](https://arxiv.org/abs/2305.13612)); for our base model, we ignore the stutter prediction components.

![](https://github.com/user-attachments/assets/35cd4b70-128a-4db7-a171-a5042240c6f9)

We made the following modifications to the base model:

- The base model uses the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) for the alignment of the mel spectrogram to the transcript. We found that the alignment timestamps of MFA are often not very precise, and we decided to use instead a modified version of WhisperX (with wav2vec2-xlsr-53-espeak-cv-ft as the aligner) because of the better alignment precision. This entailed restructuring and modifying parts of the base model to accommodate the change of aligner. 

- The base model implementation of how the regions of the original audio to be edited and how short silences are handled had some important issues, which resulted, for example, in initial/final phonemes of the edited regions being cut off if certain (fairly common) conditions occured. We instead implemented a modified version to make the handling of the regions more stable and precise. 

- We improved the precision and the reliability of the computation of the duration of the mel spectrogram and of other markers for the edited transcript by introducing a more flexible hyperparameter `mask_loc_buffer`. 

#### **The pipeline of our model**

The precise pipeline of our model is the following (a red square indicates those parts where we have modified significantly the base model, while a blue square indicates the parts where we later did single-speaker fine-tuning):

- INPUT: wav, transcript, edited transcript, and edited word regions.
- The wav file gets converted to a mel spectrogram using `librosa`; the transcript and the edited transcript get converted to [CMU phonemes](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) via vanilla `g2p_en`.
- ![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) <span style="color:red"> **The phoneme alignment of the mel spectrogram and the transcript is performed with a modifed version of WhisperX (with wav2vec2-xlsr-53-espeak-cv-ft as the aligner). This produces as output `mel2ph`, which is a matrix with as many entries as the bins of the mel spectrogram, and for each entry it has the (number corresponding to the) phoneme heard at the time corresponding to the mel spectrogram bin.**  ![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) </span >
- The fundamental frequency `f0`  over time and the unvoiced `uv` markers over time on the original audio are determined via vanilla `parselmouth`; the input to `parselmouth` is just wav.
- The utterance-level speaker embedding (a 256-dimensional vector) is determined via vanilla `resemblyzer`; the input to `resemblyzer` is just wav.
- ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) <span style="color:cyan"> **The edited phonemes (alone) are passed to `FastSpeechEncoder`; each phoneme is assigned a 256-dimensional vector. The result of this operation is `encoder_out`.** ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) </span >
- ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) <span style="color:cyan"> **The utterance-level speaker embedding from `resemblyzer` (alone) is sent to `spk_embed_proj`, which is a linear layer producing another 256-dimensional vector. If a speaker_id is used, a single fixed 256-dimensional vector is added to this. The result is `style_embed`.** ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) </span >
 - ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) <span style="color:cyan"> **We compute `dur_inp` as the sum of the vectors `style_embed` and `encoder_out`. This vector and a masked mel2ph matrix are sent to the forward duration predictor. The masking is determined by the edited transcript and edited word regions.** ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) </span >
- <span style="color:red"> ![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) **`edited_mel2ph`, `edited_f0`, `edited_uv`, and `ref_mels` are computed from the output of the duration predictor and from the edited word regions.  Here, `edited_mel2ph` is computed by sticking the predicted mel2phonmes matrix coming from the duration predictor inbetween the two `mel2ph` ends of the original region to be edited removed; `edited_f0`, `edited_uv`, and `ref_mels` are, respectively, just the original `f0`, `uv`, and mel spectrogram with zeros put in the unknown region to be edited.**
![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png)  </span >
- ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) <span style="color:cyan"> **`edited_f0`, `edited_uv`, `style_embed`, `edited_mel2ph`, and `encoder_out` are sent to the pitch prediction model. The result is concatenated with `encoder_out` and `edited_mel2ph` to determine `decoder_inp`. There is an option to pass this result into a decoder, but it is set to false for the model we are basing things off of. So for each mel bin, we have an associated f0, phoneme, and decoder output associated to that phoneme.** ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) </span >
- `decoder_inp` is summed with the result of passing the `ref_mels` (alone) to the MelEncoder to get `cond`, which is our context conditioning.
- `cond` is passed to the spectrogram denoiser to get the inferred spectrogram for the edited region, called `output`.
- `output` is concatenated within the original mel spectrogram to get the full spectrogram of the edited audio.
- The resulting full spectrogram is passed to a vocoder to produce the resulting wav of the edited audio.
- OUTPUT: the resulting wav of the edited audio.

#### **An example of speech editing**

Here is an example of the speech editing, which can be run in the Jupyter notebook speech_edit_inference_with_spk_embed.ipynb. The audio that we are going to edit has a voice saying "This is a libri vox recording". We are going to generated the same voice saying "This is a little parachuting fox recording".

```python
total_time_start = time.time()
# during inference we can choose if we want to use MFA for alignment by setting use_MFA=True here. Otherwise whisperX is used for phoneme alignment, which is much faster
use_MFA = False
# we can also choose if we want to use their method of librosa for spectrograms, or ours of torchaudio
use_librosa = False
# whether or not to save the file
save_wav_bool = False
# whether or not to print the inferred audio
disp_wav = True

# this should be a positive integer determining how many mel bins to the left to shift the end of the left part of the masked spectogram and the start of the right part of the masked spectrogram
# each mel bin corresponds to roughly 256/22050 = .0116 seconds of the original audio
# if the buffer is large enough that it overlaps with the previously detected phoneme, the midpoint of the silence before the first word to change is used instead
mask_loc_buffer = 5

# We now pass all the info about the original audio and the regions to be edited.
dataset_info = [{}]
dataset_info[0]["item_name"] = "1"  # this should just be used for naming the output file
dataset_info[0]["text"] = "this is a libri vox recording"  # a transcription of the original text
dataset_info[0]["wav_fn_orig"] = "inference/audio_backup/1.wav"  # location of the .wav file to perform inference on
dataset_info[0]["region"] = "[4,5]"  # the region to edit (counting the words that will be changed starting from 1)
dataset_info[0]["mfa_textgrid"] = ""  # we still need to set this to some value even if we are not using MFA
dataset_info[0]["edited_text"] = "this is a little parachuting fox recording"  # the full text to edit to
dataset_info[0]["edited_region"] = "[4,6]"  # word counts in the full edited text of the region which is to be inferred starting from one
# performing inference
infer_class_obj.example_run(dataset_info, use_MFA, use_librosa, save_wav_bool, disp_wav, mask_loc_buffer)

print(f"Total Time: {time.time()-total_time_start}")
```
Here are the results: [original_audio](https://github.com/user-attachments/assets/e8ba17a1-357e-4a46-89c2-714df983adac), [edited_audio](https://github.com/user-attachments/assets/d6a74f8e-6cec-4f0e-9a32-1b64304f5226).

#### **A GUI for speech editing audio files with looping**
The notebook looping_audio_speech_editing.ipynb allows the user to play around with a GUI to edit a .wav file from the user's local machine while looping, for convenience. Here's a short demo of how the GUI works: [demo_looping_audio_GUI](https://github.com/user-attachments/assets/342952cc-1f01-4379-960a-73818db96e96).

![](https://github.com/user-attachments/assets/98e85a1f-adea-4dba-94a0-0d2f8ecf47f9)




The user should input the following information:
- `audio_src` : the path to the .wav file on the user's local machine (make sure the audio has a sample rate of 16000 and 1 channel).
- `NUM_ITER` : The total number of chunks the stream will get from the input in one "background" loop. The current default length of a chunk is roughly 1.2 seconds (the length of a single chunk can be changed by editing the value of `segment_length`). So, for example, if `NUM_ITER` is 40 and if the chunk has the default length, then a "background" loop will process 48 seconds of audio.
- `key_phrases_dict_orig` : a dictionary with the words/phrases to be edited in the keys and the target words/phrases in the values. (For example, {sun: moon} will edit "sun" into "moon".)
  

In the GUI, 
- the toggle button  `Pause (background) loops`/ `Resume (background) loops` allows the user to pause or resume the background looping (in which more chunks from the input stream are processed). 
> [!WARNING]
>  The background loop and the audio playback that the user hears might not coincide. This is because the background looping is processing the audio file much faster than the time it takes for the audio to playback. The GUI shows the current background loop number; we suggest to pause the background looping frequently,  so that any dictionary updates can come into effect more quickly (they usually come into effect during the background loop after the current one). <br>

- In order to add or remove new words to be flagged and edited, the user can push the `Add Row` button and enter the new values. Pushing the `Remove Row` button will remove the last row. In order to update the dictionary, the user then needs to push the `Update Dictionary` button. Leaving both the entries of a row empty has functionally the same effect of removing that row (once the dictionary gets updated).
> [!IMPORTANT]
> The effects of the new edited dictionary might take a while to take place. This is due to the fact that, while running, the background loop is much further ahead in processing the input stream than the played back audio might seem to suggest. Usually, any dictionary updates will take place at latest during the background loop after the current one. <br>
- To close the GUI, click on the `Close` button. While the GUI window should close almost immediately, it could take a few seconds for the background processes to terminate.







#### **Ablation studies**

We performed some ablation studies, to compare how our changes affected the performance relative to the base model. We outline here some of the results.

Change 1: Modifying the way that short silences are handled. We found that in some cases this alleviates a 'silent phoneme problem.' 

Change 2: Replacing MFA with whisperX and custom wav2vec model for alignment, and substitute out some other vanilla tools being used at different stages of inference to ones that can keep things on GPU.

Our other changes, we do not perform an ablation study for. The first is to introduce a hyperparamater which changes how the inferred audio is inserted into the original audio. Since the losses are only computed in the inferred region, and the inferred region is not changing, this will not affect the losses, even though it can make the speech sound much more natural. Lastly, because even with our changes the `silent phoneme problem` persists, we have the inferred region automatically adjust until this no longer happens. Since the inferred region is changing with this modification, it isn't really fair to compare the losses, since a different inference problem is occuring.

Validation results on a subset of the [LibriTTS corpus](https://openslr.org/60/)' test set  (38 speakers, 3013 utterances) 


| Loss functions            |    f0    |   l1_coarse   |   pdur   |   ssim_coarse   |   uv   |   wdur   |   total_loss   |
|---------------------------|----------|---------------|----------|-----------------|--------|----------|----------------|
| Baseline – MFA GT | 0.2075               | 0.1956        | 0.0603                 | 0.2001            |0.7872       | 0.2286              |1.6791   |
| Change 1 – MFA GT         | 0.2075               | 0.1955         | 0.0596                | 0.1999             | 0.7891        |       0.2238| 1.6756     |
| Changes 1 + 2 – MFA GT      | 0.1942               | 0.1661         | 0.1012                 | 0.1793             | 0.8771       |   0.6287  |2.1466      |
| Changes 1 + 2 – whisperX GT      | 0.1568               | 0.1663         | 0.0669                 | 0.1778            | 0.5023       | 0.1427  | 1.2128           |

#### **Fine-tuning on individual speakers**

If the speech editor is used by a specific person, it makes sense to fine-tune some of the components of the model to that specific person.
Since the base model FluentSpeech was trained on the [LibriTTS corpus](https://openslr.org/60/), which mainly includes American speakers, we decided to pick a British female speaker with a quite distinctive voice as our specific speaker (the Narrator from Baldur's Gate 3, voiced by amazing Amelia Tyler). 
We found a 4 hours video of background-noise-free Narrator's lines on YouTube. After downloading the audio, we preprocessed it by cutting it into chunks based on silences of a certain minimal length; for each chunk, we then used Whisper to get a first tentative transcript; we then manually checked cleaned the transcripts to make sure that they matched as much as possible with the audio chunk; we then trimmed any trailing silences in the audio chunks (as it helps alignment); finally, we put the cleaned dataset into a form that could be passed to our model. For data cleaning a preprocessing, format your cleaned audio chunks and transcripts in the same way as LibriTTS, then modify the directory names in data_gen/tts/base_preprocess.py, data_gen/tts/base_binarizer.py, and config.yaml according to your dataset location and name and run base_preprocess then base_binarizer.

We then run two different types of fine-tuning: a naive one, and a more refined one. 
For the more refined type of fine-tuning, following the paradigm outlined in the recent seminal paper [AdaSpeech](https://arxiv.org/abs/2103.00993) we only fine-tuned on conditional input certain specific layers of our model involving the speaker embedding and speaker style. This allows us to fine-tune as few parameters as possible, while still hoping to ensure good adaptation quality. In our specific case, we are only fine-tuning 74496 parameters out of the 31560867 total parameters in the model, which is 0.23% of the total.

We run fine-tuning on a local machine with a GTX1080 for for 17,750(naive)/ 53,250(ada) steps. Compared to the base model losses, most of the losses went significantly down with the AdaSpeech-style fine-tuning:

| Loss functions            |    f0    |   l1_coarse   |   pdur   |   ssim_coarse   |   uv   |   wdur   |   total_loss   |
|---------------------------|----------|---------------|----------|-----------------|--------|----------|----------------|
| Baseline (no fine-tuning)  | 0.2072               | 0.2393         | 0.1128                 | 0.2545             | 1.1229        | 0.1465               | 2.0832                  |
| Naïve fine-tuning          | 0.1781               | 0.1993         | 0.0868                 | 0.2187             | 0.5680        | 0.0889               | 1.3397                  |
| AdaSpeech fine-tuning      | 0.1482               | 0.1487         | 0.0460                 | 0.1670             | 0.3653        | 0.0415               | 0.9167                  |


What is interesting to notice is that, while the finetuned voice resembles better the ground truth voice, the phonemes enunciation becomes worse.
Here's an example of a ground truth audio with some parts masked and then reconstructed using the AdaSpeech fine-tuned model:
[ground_truth_audio](https://github.com/user-attachments/assets/5b7d927b-b024-4d83-a114-ebd2c84c0e86), [reconstructed_audio](https://github.com/user-attachments/assets/c0973468-f246-4316-86cc-cd655b4114b1).


 We speculate that this is due to the difference between phonemes pronounced with an American accent and phonemes pronounced with a British accent. In the base model, both the encoder and decoder are trained on mainly American speakers. With our fine-tuning, we have changed the weights of the encoder, but not those of the decoder (for lack of time, because we didn't have access to the base model's decoder weights). In future work, we plan to train the decoder from scratch on specific speakers to see if it helps with the phonemes enunciation issue. 

After fine-tuning AdaSpeech-style, we also performed an ablation study by reverting back to their non-fine-tuned values different combinations of groups of weights, to better understand the effect of fine-tuning. The following picture shows the total loss on the validation set for the different combinations; each letter appearing in a combination represents a different group of weights for which we used the fine-tuned value (if a letter doesn't appear, it means that for that group of weights we reverted to the non-fine-tuned values). The meaning of the letters are:
- e: speaker embed layer normalisations
- l: duration predictor layer normalisations
- p: pitch predictor layer normalisations
- s: utterance-level speaker embed linear layer
- i: speaker id weight
  
![image](./ablft.png)

You can run our naive fine-tuning method on your processed dataset via: 

```
python tasks/run.py --config checkpoints/spec_denoiser/config.yaml --exp_name spec_denoiser --naive_fine_tune
```
and our ada-speech style fine-tuning method via: 

```
python tasks/run.py --config checkpoints/spec_denoiser/config.yaml --exp_name spec_denoiser --ada_fine_tune
```

 Make sure you have modified config.yaml to have use_spk_id=True and num_spk=1. Set max_updates according to how many steps you want to take before the code stops executing (this is total steps, not additional steps, so if you set it to less than however many steps have been taken on the original model you are fine tuning on, nothing will happen). Also, if you have downloaded our pretrained checkpoints, remove them from checkpoints/spec_denoiser, keeping only the original model there.

 If you are curious to see our process for selecting the parameters to fine-tune and more exposition, feel free to take a look at looking_at_model_params.ipynb.


## 2. Implementing a live streaming speech editing demo

Here's a link to a demo video of our live streaming speech editing app: [demo video](https://www.youtube.com/watch?v=k9yhQQq4Tew)

The Jupyter notebook with the implementation of the live streaming speech editing demo is streaming_proof_of_concept.ipynb.

> [!WARNING] 
> The Jupyter notebook and the following information was only tested on Windows. <br>

After selecting the audio and video sources for the streaming (follow the instructions at the top of the notebook), the app starts a Tkinter GUI. Within the Tkinter mainloop, three or four threads are called, depending on whether video is played or not. The first thread deals with the input stream. The second thread deals with processing the stream input for a quick transcription (using the model "distil-whisper/distil-large-v3") and inference using our model (in the case flagged words are found in the transcription). While the GUI is running, this thread accumulates chunks from the input stream and uses the package Silero to see if it detects speech or not and, if yes, how many portions of speech it detects. Once certain conditions involving the number of speech portions and silences detected are met, we cut the segment of chunks accumulated so far at a particular silence and send it for transcription (and potentially inference); after the transcription and/or inference has occured, we send the results to the third thread and we then start accumulating chunks from the input stream again. The third thread deals with playing back the (potentially) edited audio, together with outputting on the Tkinter GUI the spectrograms and characters alignment of the current audio segment, its transcription, and video frames (if `play_video` is set to **True**). A fourth thread is called if video is to be played; this thread waits for the third thread to update the GUI for the first time to start playing the video in the GUI.


#### **User input**
In order for the demo to run, the user needs to input some information:

- `inp_device` : this should be set to **None** if the audio stream will come from a .wav file audio saved on the local machine; <br>
it should be set to **dshow** if the audio stream will come from the user's microphone.
> [!WARNING]
> For Mac or Linux users, something other than **dshow** might need to be used. See the instructions available [here](ttps://pytorch.org/audio/main/tutorials/device_asr.html) <br>

- `audio_src` : this should be set to the path to the user's .wav file (e.g. '.\test_audio.wav') if the audio stream will come from that file;
> [!NOTE]
> The .wav file should have a sample rate of 16000 and should be mono channel. You can easily conver a .wav audio file to these specifications by running the following code
> ```python
> from pydub import AudioSegment
> input_file_path = "path/to/your/file.wav"
> converted_audio = AudioSegment.from_wav(input_file_path)
> converted_audio = converted_audio.set_frame_rate(16000).set_channels(1)
> output_file_path = "path/to/your/output_file.wav"
> converted_audio.export(output_file_path, format="wav")
> ``` 
<br>

> [!NOTE]
> In the current implementation of the demo, the option of using a YouTube link means that the program will automatically download both the audio and video from the YouTube url and save them in the correct format in the user's local machine. <br>

it should be set to the user's microphone ID if the audio stream will come from the user's microphone. In order to find out their device ID, the user can open the command prompt and type 
```
ffmpeg -list_devices true -f dshow -i dummy
```
> [!WARNING]
> The command will be different for Mac or Linux users. <br>

After identifying the wanted device, the user should use the alternative name form of the device ID and set `audio_src` to 
```python
'audio=' + ALTERNATIVE_NAME_DEVICE_ID
```

- `play_video`: this should be set to **False** if no video need to be played; <br>
it should be set to **True** if the audio comes with a video and the user wants to play the video as well.
- `video_source`: this could be set to **None** (or to whatever) if  `play_video` was set to **False**; <br>
it should be set to the path to the user's video .mp4 file (e.g '.\test_video.mp4') if the video file comes from the user's local machine (this is the case also for the option of using a YouTube url in the current implementation); <br>
it should be set to 0 if the video source is the user's cam.

> [!TIP]
> Some additional advanced settings that the user might want to modify:
>
> - `DO_RUBBERBAND`: We implemented an option to do "rubberbanding" - that is, speed up/slow down a chunk of audio to catch up on the live stream (especially after inference). When using rubberbanding, the audio output will stay at a fixed delay from the start time of the stream (usually well under 10s). Rubberbanding is particularly useful when the user wants the audio (and video) playback to be always at a fixed delay from the live streaming.
<br>


#### **How to use the GUI**

![](./gui.png)

- The toggle button  `Pause (background) loops`/ `Resum (background) loops` functionally doesn't do anything here (it is used in the GUI for editing audio while looping -- see a previous section).
- In order to add or remove new words to be flagged and edited, the user can push the `Add Row` button and enter the new values. Pushing the `Remove Row` button will remove the last row. In order to update the dictionary, the user then needs to push the `Update Dictionary` button. Leaving both the entries of a row empty has functionally the same effect of removing that row (once the dictionary gets updated).
> [!IMPORTANT]
> The effects of the new edited dictionary might take a bit to take place.  <br>
- To close the GUI, click on the `Close` button. While the GUI window should close almost immediately, it could take a few seconds for the background processes to terminate.



## 3. Future work

- Properly clean the code (there are lots of vestigial remains from the *FluentSpeech* code that we don't actually use).
- Conditionally fine-tune the phonemes decoder (instead of just the encoder) as well, in order to improve the quality of the generated edited speech when tailoring our model to an individual speaker, in order to solve some phoneme enunciation problems encountered.
- Integrate better the various models used (we probably don't need all these models!)
- Improve the stability and expand the range of functionalities of the live-streaming speech editing app.


## 4. Installation requirements and quick inference on audio files

#### **Requirements to run our model**

The first step is to install the [cuda toolkit](https://developer.nvidia.com/cuda-downloads) if you have a cuda-enabled gpu. 

Then you should install [pytorch](https://pytorch.org/get-started/locally/) depending on the cuda version you have.

Following this step, the necessary packages and their versions can be found in `requirements.txt`. It is possible to install these via 
```
pip install -r requirements.txt
```
 once the repository is cloned and pulled. In addition, you will need to download config.yaml and xxx.ckpt from the `Pretrained Checkpoint` section, model_ckpt_steps_2168000.ckpt and config.yaml from the `Download the pre-trained vocoder` section, and phone_set.json,word_set.json from the `Tips` section in FluentSpeech's [github repository](https://github.com/Zain-Jiang/Speech-Editing-Toolkit). The pretrained checkpoint from their model should go in checkpoints/spec_denoiser, the vocoder in pretrained/higan_hifitts, and phone_set.json,word_set.json in data/processed/binary/libritts. 

 You will then need to download [ESpeakNG](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#installation). 

 Now, you should be able to open speech_edit_inference.ipynb. In the `user defined directories' section, set your Espeak_dll_directory to the correct location. Upon first inference, a whisperXmodel will automatically be downloaded. If you want to control where this goes, set the whisperX_model_directory. Don't worry about the MFA paths unless you want to use the original model's alignment, in which case you will need to [install MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/).

 At this point the examples in speech_edit_inference.ipynb should run. You can use your own files by modifying the dataset info in the `Several examples on different speakers without any fine-tuning' to your needs.


#### **Requirements to run the live streaming speech editing demo**

In order to run the live streaming speech editing demo streaming_proof_of_concept.ipynb, you need to have installed both all the requirements for our model (see the previous section) and the packages that can be found in `requirements_streaming.txt`. You will want to make sure you are using a cuda-enabled GPU (test this by running torch.cuda.is_available()). Moreover, you will need to make sure that ffmpeg is installed in your local machine and that the path to the `ffmpeg.exe` file is added to the system PATH environment variable. In order to use the Rubber Band Library, you will also need to install the [Rubber Band command-line utility](https://breakfastquay.com/rubberband/) and add the path to the folder containing the `rubberband.exe` and the `sndfile.dll` files to the system PATH environment variable.

If necessary, modify the Espeak_dll_directory and model checkpoint directories in the `User input and hyperparameters` section of the notebook as in speech_edit_inference.ipynb. 

Follow the instructions in the first few cells of the notebook depending on what audio/video source you want to run livestream editing on. On first run, a few additional models will need to be downloaded.
