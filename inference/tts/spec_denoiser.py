import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
import sys
import shutil
import pandas as pd
from tqdm import tqdm

from data_gen.tts.base_preprocess import BasePreprocessor
from inference.tts.base_tts_infer import BaseTTSInfer
from inference.tts.infer_utils import get_align_from_mfa_output, extract_f0_uv
from modules.speech_editing.spec_denoiser.spec_denoiser import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.spec_aug.time_mask import generate_time_mask
from utils.text.text_encoder import is_sil_phoneme
from resemblyzer import VoiceEncoder
from utils.audio import librosa_wav2spec
from inference.tts.infer_utils import get_words_region_from_origintxt_region, parse_region_list_from_str

import time 


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins'],hparams=hp),
}

from utils.commons.hparams import set_hparams
from utils.commons.hparams import hparams as hp
from utils.audio.io import save_wav
from utils.plot.plot import plot_mel

from utils.commons.hparams import set_hparams
from modules.vocoder.hifigan.hifigan import HifiGanGenerator

from transformers import AutoProcessor, AutoModelForCTC
import logging

import torchaudio

import IPython.display as ipd

from utils.audio.align import mel2token_to_dur

class SpecDenoiserInfer():
   
    def __init__(self, hparams,binary_data_directory,orig_ckpt_path,whisperX_model_directory, device=None,fine_tuned=None,fine_tune_ckpt_path='',ada_weights=''):

        #use the gpu if cuda is available 
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #device = 'cpu'
        self.device = device

        #hparams are set when instantiating the class. We set these by calling set_hparams(exp_name='spec_denoiser'), which loads in the args to a dictionary from checkpoints/spec_denoiser/config.yaml 
        self.hparams = hparams
        
        #location of the phoneme set, speaker mapping, and word set dictionaries for libritts from the Fluentspeech GitHub under tips
        self.data_dir = binary_data_directory

        #builds a TokenTextEncoder (utils/text/text_encoder.py) object from the word and phoneme list .json files supplied at  binary_data_directory
        self.preprocessor = BasePreprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict(self.data_dir)
        #loads the speaker map .json file
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)

        if fine_tuned=='ada':
            self.fine_tune_checkpoint_path=fine_tune_ckpt_path
            self.fine_tuning_param_names=self.get_single_speaker_ada_fine_tune_list(ada_weights)
        elif fine_tuned=='naive':
            self.fine_tune_checkpoint_path=fine_tune_ckpt_path
            self.fine_tuning_param_names='fs.spk_id_proj.weight'
        else:
            self.fine_tune_checkpoint_path=''
        
        #loads the gaussian diffusion model. This will be used in the "spectrogram denoising" step of inference
        self.model = self.build_model(orig_ckpt_path,self.fine_tune_checkpoint_path,fine_tuned)
        self.model.eval()
        self.model.to(self.device)

        #loads the vocoder model. This is used at the last step of inference in order to convert the spectrogram output from the gaussian diffusion model to audio
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

        #loads weights for a small NN build into resemblyzer which takes in a mel spectrogram and outputs the speaker embedding from pretrained.pt
        self.spk_embeding = VoiceEncoder(device=device)

        #loads the wav2vec model used for phoneme alignment with whisperX located at whisperX_model_directory. This will be downloaded from huggingface if no model is present
        self.whisper_processor,self.whisper_align_model = self.build_whisper(whisperX_model_directory)
        self.whisper_align_model.eval()
        self.whisper_align_model.to(self.device)


            


    def build_whisper(self,whisperX_model_directory):
        logging.getLogger('transformers').setLevel(logging.ERROR)
        whisperX_load_start_time=time.time()
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft",cache_dir=whisperX_model_directory)
        align_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft",cache_dir=whisperX_model_directory)
        print(f'WHISPERX LOAD TIME = {time.time()-whisperX_load_start_time}')

        
        return processor, align_model
    
    def get_single_speaker_ada_fine_tune_list(self,ada_weights):

        encoder_layer_norms = [
        "fs.encoder.res_blocks.0.blocks.0.0.weight",
        "fs.encoder.res_blocks.0.blocks.0.0.bias",
        "fs.encoder.res_blocks.0.blocks.1.0.weight",
        "fs.encoder.res_blocks.0.blocks.1.0.bias",
        "fs.encoder.res_blocks.1.blocks.0.0.weight",
        "fs.encoder.res_blocks.1.blocks.0.0.bias",
        "fs.encoder.res_blocks.1.blocks.1.0.weight",
        "fs.encoder.res_blocks.1.blocks.1.0.bias",
        "fs.encoder.res_blocks.2.blocks.0.0.weight",
        "fs.encoder.res_blocks.2.blocks.0.0.bias",
        "fs.encoder.res_blocks.2.blocks.1.0.weight",
        "fs.encoder.res_blocks.2.blocks.1.0.bias",
        "fs.encoder.res_blocks.3.blocks.0.0.weight",
        "fs.encoder.res_blocks.3.blocks.0.0.bias",
        "fs.encoder.res_blocks.3.blocks.1.0.weight",
        "fs.encoder.res_blocks.3.blocks.1.0.bias",
        "fs.encoder.last_norm.weight",
        "fs.encoder.last_norm.bias"
        ]

        decoder_layer_norms = [
            "fs.decoder.res_blocks.0.blocks.0.0.weight",
            "fs.decoder.res_blocks.0.blocks.0.0.bias",
            "fs.decoder.res_blocks.0.blocks.1.0.weight",
            "fs.decoder.res_blocks.0.blocks.1.0.bias",
            "fs.decoder.res_blocks.1.blocks.0.0.weight",
            "fs.decoder.res_blocks.1.blocks.0.0.bias",
            "fs.decoder.res_blocks.1.blocks.1.0.weight",
            "fs.decoder.res_blocks.1.blocks.1.0.bias",
            "fs.decoder.res_blocks.2.blocks.0.0.weight",
            "fs.decoder.res_blocks.2.blocks.0.0.bias",
            "fs.decoder.res_blocks.2.blocks.1.0.weight",
            "fs.decoder.res_blocks.2.blocks.1.0.bias",
            "fs.decoder.res_blocks.3.blocks.0.0.weight",
            "fs.decoder.res_blocks.3.blocks.0.0.bias",
            "fs.decoder.res_blocks.3.blocks.1.0.weight",
            "fs.decoder.res_blocks.3.blocks.1.0.bias",
            "fs.decoder.last_norm.weight",
            "fs.decoder.last_norm.bias"
        ]

        dur_predictor_layer_norms = [
            "fs.dur_predictor.conv.0.2.weight",
            "fs.dur_predictor.conv.0.2.bias",
            "fs.dur_predictor.conv.1.2.weight",
            "fs.dur_predictor.conv.1.2.bias",
            "fs.dur_predictor.conv.2.2.weight",
            "fs.dur_predictor.conv.2.2.bias"
        ]

        pitch_predictor_layer_norms = [
            "fs.pitch_predictor.conv.0.2.weight",
            "fs.pitch_predictor.conv.0.2.bias",
            "fs.pitch_predictor.conv.1.2.weight",
            "fs.pitch_predictor.conv.1.2.bias",
            "fs.pitch_predictor.conv.2.2.weight",
            "fs.pitch_predictor.conv.2.2.bias",
            "fs.pitch_predictor.conv.3.2.weight",
            "fs.pitch_predictor.conv.3.2.bias",
            "fs.pitch_predictor.conv.4.2.weight",
            "fs.pitch_predictor.conv.4.2.bias"
        ]

        speaker_embedding_weights = [
            "fs.spk_embed_proj.weight",
            "fs.spk_embed_proj.bias"
        ]

        if ada_weights=='':
            all_weights_to_fine_tune_names=encoder_layer_norms+decoder_layer_norms+dur_predictor_layer_norms+pitch_predictor_layer_norms+speaker_embedding_weights
        else:
            selection =list(ada_weights)
            all_weights_to_fine_tune_names=[]
            if 'e' in selection:
                all_weights_to_fine_tune_names=all_weights_to_fine_tune_names+encoder_layer_norms
            if 'd' in selection:
                all_weights_to_fine_tune_names=all_weights_to_fine_tune_names+decoder_layer_norms
            if 'l' in selection:
                all_weights_to_fine_tune_names=all_weights_to_fine_tune_names+dur_predictor_layer_norms
            if 'p' in selection:
                all_weights_to_fine_tune_names=all_weights_to_fine_tune_names+pitch_predictor_layer_norms
            if 's' in selection:
                all_weights_to_fine_tune_names=all_weights_to_fine_tune_names+speaker_embedding_weights
        if self.hparams['use_spk_id']:
            all_weights_to_fine_tune_names=all_weights_to_fine_tune_names+['fs.spk_id_proj.weight']
        return all_weights_to_fine_tune_names
    
    def build_model(self,orig_ckpt_path,fine_tune_checkpoint_path,fine_tuned):
        load_diffusion_model_time_start=time.time()
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=self.hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[self.hparams['diff_decoder_type']](self.hparams),
            timesteps=self.hparams['timesteps'], time_scale=self.hparams['timescale'],
            loss_type=self.hparams['diff_loss_type'],
            spec_min=self.hparams['spec_min'], spec_max=self.hparams['spec_max'],hparams=self.hparams
        )
        orig_checkpoint=torch.load(orig_ckpt_path, map_location='cpu') #original checkpoint path
        if (fine_tuned=='naive') or (fine_tuned=='ada'):
                checkpoint=torch.load(fine_tune_checkpoint_path, map_location='cpu')
                model.load_state_dict(orig_checkpoint['state_dict']['model'],strict=False)
                dict_to_load={k:v for k,v in checkpoint['state_dict'].items() if k in self.fine_tuning_param_names}
                print('loading fine tuned parameters:')
                print(dict_to_load.keys())
                model.load_state_dict(dict_to_load,strict=False)
        else:
            model.load_state_dict(orig_checkpoint['state_dict']['model'])
        
                
        
        
        model.to(self.device)
        model.eval()
        print(f'LOAD DIFFUSION MODEL TIME: {time.time()-load_diffusion_model_time_start}')
        return model

    
    def forward_model(self, inp,mask_loc_buffer,sil_frames,fix_silent_phonemes):

        #extract_feat_and_mask_start_time=time.time()

        #Convert output of preprocess_input to tensors for input into diffusion mode. 
        #Also gets speaker embedding using the original loaded .wav file from the resemblyzer model instantiated as self.spk_embeding
        #The speaker embedding is the only additional thing generated here
        sample = self.input_to_batch(inp)

        
        edited_txt_tokens = sample['edited_txt_tokens']
        mel = sample['mel']
        mel2ph = sample['mel2ph']
        mel2word = sample['mel2word']
        dur = sample['dur']
        ph2word = sample['ph2word']
        edited_ph2word = sample['edited_ph2word']
        f0 = sample['f0']
        uv = sample['uv']
        words_region = sample['words_region']
        edited_words_region = sample['edited_words_region']
        text = sample['text']

        #print('')
        #print('mel2ph:')
        #print(mel2ph)
        #print('')
        #print('uv:')
        #print(uv)
        #print('')
        #print('Silence regions from uv')
        #print(get_silence_regions_from_uv(inp['uv']))
        #print('')
        #print('words_region')
        #print(words_region)
        #print('')
        #print('edited_words_region')
        #print(edited_words_region)
        #print('')
        #print('Silence regions from uv')
        #print(get_silence_regions_from_uv(inp['uv']))
        #print('')
        #print('ph2word')
        #print(ph2word)
        #print('')
        #print('edited_ph2word')
        #print(edited_ph2word)
        #print('')

        #[int1,int2] with int1, int2 counting the number of words until the first and last word to change, including silences 
        good_ph_len_flag=False
        edited_word_idx = words_region[0]
        changed_idx = edited_words_region[0]

        # Forward the edited txt to the FastSpeechEncoder - assigns an embedding of size 256 to each phoneme in edited_txt_tokens
        encoder_out = self.model.fs.encoder(edited_txt_tokens)  # [B, T, C]

        #in edited_txt_tokens, 0 is reserved for "<pad>", so this gets 0's for '<pad>' and 1 otherwise
        src_nonpadding = (edited_txt_tokens > 0).float()[:, :, None]
            
        #gives the speaker embedding to fastspeech, returns another tensor of size 256
        style_embed = self.model.fs.forward_style_embed(sample['spk_embed'], sample['spk_ids'])

        #print('Style Embedding:')
        #print(style_embed)
        
        #I think if we want to specialize to a speaker we just do + voice_encoder.embed_speaker(wavs) to style embed above, where wavs are a bunch of cleaned audio files of that speaker and voice endcoder is from resemblyzer.
        #Or possibly we apply the spk_id_proj from fs to this value before adding


        dur_inp = (encoder_out + style_embed) * src_nonpadding
        
        #these correspond to end of sentence, punctuation, beginning of sentence, and spaces. Not sure what 2 is???
        sil_tokens=set([i for i in range(1,10)]+[79])
        
        #sometimes the model will predict that certain phonemes should not be uttered at all, or only for one frame. Here, we adjust the word region to the left until this doesn't happen any more.
       
        while good_ph_len_flag==False:                
            ret = {}            
            #masked_dur holds duration (in terms of number of mel bins) for phonemes which are not changed from the original, and 0's for those that will be inferred
            masked_dur = torch.zeros_like(edited_ph2word).to(self.device)
            masked_dur[:, :ph2word[ph2word<edited_word_idx[0]].size(0)] = dur[:, :ph2word[ph2word<edited_word_idx[0]].size(0)]
            if ph2word.max() > edited_word_idx[1]:
                masked_dur[:, -ph2word[ph2word>edited_word_idx[1]].size(0):] = dur[:, -ph2word[ph2word>edited_word_idx[1]].size(0):]                
            # Forward duration model to get the duration and mel2ph for edited text seq (Note that is_editing is set as False to get edited_mel2ph)
            #Z: They say this, but there is no is_editing bool that I can see
            #generating inputs for fastspeech duration model                
            masked_mel2ph = mel2ph.clone() #SEARCH FOR THIS - JUST ADDED, NOT SURE IF NECESSARY                
            masked_mel2ph[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])] = 0
            time_mel_masks_orig = torch.zeros_like(mel2ph).to(self.device)
            time_mel_masks_orig[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])] = 1.0
            #use fastspeech model to predict how long each phoneme in the edited speech will be held for. Returns string of ints with same meaning as mel2ph                 
            edited_mel2ph = self.model.fs.forward_dur(dur_inp, time_mel_masks_orig, masked_mel2ph, edited_txt_tokens, ret, masked_dur=masked_dur, use_pred_mel2ph=True)
            #duration of the phonemes in edited txt_tokens
            #edited txt tokens contains tokens corresponding to each phoneme in the edited text
            #1=<BOS>, 8=<EOS>,79= |                 
            #the duration of the tokens in the region which is being edited
            edited_dur=mel2token_to_dur(edited_mel2ph,len(inp['edited_ph_token'])).squeeze(0)[(edited_ph2word[0]>=changed_idx[0]) & (edited_ph2word[0]<=changed_idx[1])]                
            #print('edited_dur:')
            #print(edited_dur)
            #print(np.shape(edited_dur))
            #print('')
            #print('tokens:')
            #print(edited_txt_tokens.squeeze(0)[(edited_ph2word[0]>=changed_idx[0]) & (edited_ph2word[0]<=changed_idx[1])])
            #print('')
            #print('too short tokens:')
            #print(edited_txt_tokens.squeeze(0)[(edited_ph2word[0]>=changed_idx[0]) & (edited_ph2word[0]<=changed_idx[1])][edited_dur<2])
            #print('')                
            #print('')
            #print('original edited_mel2ph:')
            #print(edited_mel2ph)
            #print('')
            #print('the too short length phoneme set:')
            #print(set((edited_txt_tokens.squeeze(0)[(edited_ph2word[0]>=changed_idx[0]) & (edited_ph2word[0]<=changed_idx[1])][edited_dur<2]).to('cpu').numpy()))
            #print('')
            #if there is a token which is predicted to be uttered for a duration of less than two frames in the edited text, then adjust the edited word region by -1 (effectively -2 because spaces are counted as words)
            if not fix_silent_phonemes:
                good_ph_len_flag=True 
            elif (not set((edited_txt_tokens.squeeze(0)[(edited_ph2word[0]>=changed_idx[0]) & (edited_ph2word[0]<=changed_idx[1])][edited_dur<2]).to('cpu').numpy()).issubset(sil_tokens)) and (edited_word_idx[0]>2):
                edited_word_idx[0] -=2
                changed_idx[0] -= 2
                print('Silent inferred phonemes predicted, changing word regions!')
            else:
                good_ph_len_flag=True  


            

        #print('')
        #print('original edited_mel2ph:')
        #print(edited_mel2ph)
        #print('')
        
        #get how long each word is said for from the prediction of how long each phoneme is said for 

        edited_mel2word = torch.Tensor([edited_ph2word[0][p - 1] for p in edited_mel2ph[0]]).to(self.device)[None, :]

        #get what the index of the first and last edited phonemes are in mel2word 
        head_idx = mel2word[mel2word<edited_word_idx[0]].size(0)
        right_part_start=mel2word[mel2word<=edited_word_idx[1]].size(0)
        
        #frame buffer implementation (this can probably be improved)
        def shift_mask_loc_change(orig_left_part_start_frame,orig_right_part_start_frame,mask_loc_buffer,sil_frames):
            #orig_left_part_start_frame,orig_right_part_start_frame are the frames determining the mask position via mel2word
            #mask_loc_buffer is a user-input hyperparameter determining how many frames to the left to shift their frame position
            #sil_frames holds a list of the form [[int1,int2],[int3,int4],...] with each element [intk,int(k+1)] corresponding to an interval of frame numbers detected to contain a silence via get_align_from_mfa_output
            #outputs what numbers to add for the buffer by testing if it is more than half the way into the preceeding silence interval
            if mask_loc_buffer==0:
                return 0,0
            max_shift_dist_left_part=0
            max_shift_dist_right_part=0
            for i,frame in enumerate(sil_frames):
                if (frame[0]<=orig_left_part_start_frame)and(frame[1]>=orig_left_part_start_frame):
                    max_shift_dist_left_part=(orig_left_part_start_frame-frame[0])//2
                elif (frame[0]<=orig_right_part_start_frame)and(frame[1]>=orig_right_part_start_frame):
                    max_shift_dist_right_part=(orig_right_part_start_frame-frame[0])//2
                    break
            if max_shift_dist_left_part==0:
                print('WARNING: The masking location for the left side of the spec is not during an interval of silence.')
                left_part_end_change=0
            elif max_shift_dist_left_part<mask_loc_buffer:
                print(f'Mask loc buffer set to {mask_loc_buffer} frames, but there are only {2*max_shift_dist_left_part} frames of silence before the first edited word. Using silence midpoint instead.')
                left_part_end_change=max_shift_dist_left_part
            else: 
                left_part_end_change=mask_loc_buffer

            if max_shift_dist_right_part==0:
                print('WARNING: The masking location for the right side of the spec is not during an interval of silence. This is not a problem if you are replacing the last word in the original audio.')
                right_part_start_change=0
            elif max_shift_dist_right_part<mask_loc_buffer:
                print(f'Mask loc buffer set to {mask_loc_buffer} frames, but there are only {2*max_shift_dist_right_part} frames of silence after the last edited word. Using silence midpoint instead.')
                right_part_start_change=max_shift_dist_right_part
            else: 
                right_part_start_change=mask_loc_buffer
            return left_part_end_change,right_part_start_change 
        
        #print('sil_frames:')
        #print(sil_frames)
        #print('sil_frame times in sec:')
        #print([[256/22050*frame[0],256/22050*frame[1]] for frame in sil_frames])


        #subtract the buffer or midpoint if buffer is too large
        left_part_end_change,right_part_start_change =shift_mask_loc_change(head_idx,right_part_start,mask_loc_buffer,sil_frames)
        head_idx-=left_part_end_change
        right_part_start-=right_part_start_change 

        


        #how many additional mel bins need to be generated - compares the side of the "middle" part of edited_mel2word and mel2word
        #length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0) - mel2word[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])].size(0)
        length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0)-(right_part_start-head_idx)

        #will replace edited_mel2ph
        edited_mel2ph_ = torch.zeros((1, mel2ph.size(1)+length_edited)).to(self.device)

        tail_idx =right_part_start  + length_edited

        #print('')
        #print('head_idx')
        #print(head_idx)
        #print('')
        #print('tail_idx')
        #print(tail_idx)
        #print('')
        #print('right_part_start_change')
        #print(right_part_start_change)
        #print('')
        #print('left_part_end_change')
        #print(left_part_end_change)
        #print('')
        #print('length_edited')
        #print(length_edited)
        #print('')
        #print('mel2ph[mel2word>edited_word_idx[1]].min()')
        #print(mel2ph[mel2word>edited_word_idx[1]].min())
        #print('')
        #print('edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max()')
        #print(edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max())
        #print('')



        #fill in the original values from mel2ph for the beginning of edited_mel2ph and use the preditictions from the fastspeech model for the middle. Supposedly this is not what the fastspeech model does by default?
        edited_mel2ph_[:, :head_idx] = mel2ph[:, :head_idx] #index 0 to head_idx-1
        edited_mel2ph_[:, head_idx:tail_idx] = edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])] #index head_idx to tail_idx-1

        #mel2word.max() is the number of words (inluding silences) in the original text. 
        #we should always have mel2word.max()>edited_word_idx[1], since even if we edit up to the last word, there is a trailing silence
        #in the case mel2word.max()>edited_word_idx[1]+1, we shift everything to zero then add back in the index corresponding to the last non-silent phoneme in the inferred text

        #mel2ph[:,right_part_start+right_part_start_change:] always starts at mel2ph[mel2word>edited_word_idx[1]].min(), since right_part_start+right_part_start_change=right_part_start=mel2word[mel2word<=edited_word_idx[1]].size(0)
        #this could correspond to a silent phoeneme or a voiced phoneme
        #edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() is the maximum value added into the middle. It will always be a voiced phoneme
        #the change from the middle to the start of the right part should be the same as the change from mel2ph[:,right_part_start+right_part_start_change-1] to mel2ph[:,right_part_start+right_part_start_change]
        #that is, 1 if the last non-silent phoeneme in edited_word_idx[1] is followed by a silent phoneme in mel2ph and 2 if it is followed by a voiced one
        #note that mel2ph[:,right_part_start+right_part_start_change-1]=mel2ph[mel2word<=edited_word_idx[1]].max()
        #so we should replace edited_mel2ph_[:, tail_idx+right_part_start_change:] by mel2ph[:,right_part_start+right_part_start_change:]-mel2ph[mel2word>edited_word_idx[1]].min() (make it start at zero) 
        #+edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() (shift to start at the same value that the edited region ends at )
        #+ mel2ph[mel2word>edited_word_idx[1]].min() -mel2ph[mel2word<=edited_word_idx[1]].max()
        # which simplifies to  mel2ph[:,right_part_start+right_part_start_change:] +edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() -mel2ph[mel2word<=edited_word_idx[1]].max()
        #we also need to change masked_mel2ph = mel2ph to masked_mel2ph = mel2ph.clone() since the masking was also changing mel2ph with the former


        #if we added a buffer, we assign that buffer region to the silent phoneme inbetween these
        #if mel2word.max()=edited_word_idx[1]+1, then we just assign the 'right side' of mel2ph to be the final silent <'EOS'> phoneme

        if mel2word.max() > edited_word_idx[1]+1:            
            edited_mel2ph_[:, tail_idx+right_part_start_change:] = mel2ph[:,right_part_start+right_part_start_change:] + edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max()-mel2ph[mel2word<=edited_word_idx[1]].max()
            edited_mel2ph_[:, tail_idx:tail_idx+right_part_start_change] = (edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max())*torch.ones_like(edited_mel2ph_[:, tail_idx:tail_idx+right_part_start_change])+1 
            #edited_mel2ph_[:, tail_idx:] = mel2ph[mel2word>edited_word_idx[1]] - mel2ph[mel2word>edited_word_idx[1]].min() + edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() + 2
        else:
            edited_mel2ph_[:, tail_idx:]=(edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max())*torch.ones_like(edited_mel2ph_[:,tail_idx:])+1
        edited_mel2ph = edited_mel2ph_.long()
        

        #this is repeated from above and only edited_mel2ph have changed, not mel2word, so it seems unnecessary
        #length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0) - mel2word[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])].size(0)
        #head_idx = mel2word[mel2word<edited_word_idx[0]].size(0)
        #tail_idx = mel2word[mel2word<=edited_word_idx[1]].size(0) + length_edited

        #print(f'Mask created from bin {head_idx} (about {head_idx*256/22050} seconds) to bin {right_part_start} (about {right_part_start*256/22050} seconds)')
        #print('')
        #print('masked mel2ph:')
        #print(mel2ph)
        #print('')
        #print('edited_mel2ph')
        #print(edited_mel2ph)
        #print('')
        #print('edited_txt_tokens')
        #print(edited_txt_tokens)
        #print('')
        #print('edited_txt_tokens size:')
        #print(edited_txt_tokens.size())
        #print('')
        # Create masked ref mel by concating the head and tail of the original mel
        ref_mels = torch.zeros((1, edited_mel2ph.size(1), mel.size(2))).to(self.device)
        T = min(ref_mels.size(1), mel.size(1))
        ref_mels[:, :head_idx, :] = mel[:, :head_idx, :]
        #ref_mels[:, tail_idx:, :] = mel[mel2word>edited_word_idx[1]]
        ref_mels[:, tail_idx:, :] = mel[:,right_part_start:,:]

        # Get masked frame-level f0 and uv (pitch info)
        edited_f0 = torch.zeros((1, edited_mel2ph.size(1))).to(self.device)
        edited_uv = torch.zeros((1, edited_mel2ph.size(1))).to(self.device)
        edited_f0[:, :head_idx] = f0[:, :head_idx]
        #edited_f0[:, tail_idx:] = f0[mel2word>edited_word_idx[1]]
        edited_f0[:, tail_idx:] = f0[:,right_part_start:]
        edited_uv[:, :head_idx] = uv[:, :head_idx]
        #edited_uv[:, tail_idx:] = uv[mel2word>edited_word_idx[1]]
        edited_uv[:, tail_idx:] = uv[:,right_part_start:]
        # Create time mask - 1s on mel bins in the middle that will be replaced and 0s otherwise
        time_mel_masks = torch.zeros((1, edited_mel2ph.size(1), 1)).to(self.device)
        time_mel_masks[:, head_idx:tail_idx] = 1.0
        
        #start_get_output_time=time.time()
        #print(f'Extract Feat and Mask Time {start_get_output_time-extract_feat_and_mask_start_time} - there seems to be a lot of transferring between gpu and cpu here that we can fix. There is also some inference happening that we might be able to optimize, like the speaker embedding')

        with torch.no_grad():
            #passes to forward in GaussianDiffusion
            #there txt_tokens, time_mel_masks, mel2ph, spk_embed, f0, and uv are passed to the FastSpeech model, which returns the decoder input 
            #this does self.encoder(txt_tokens) again for some reason 
            #also does self.forward_style_embed(spk_embed) (which allows for speaker id!) again 
            #and self.forward_dur again
            #since use_pitch_embed is true, this is the only part which we haven't done already that the FastSpeech model is doing here to get decoder_inp, then self.mel_encoder(ref_mels*(1-time_mel_masks)) is used for the first time and added to decoder_inp 
            #to make cond. cond is the only input other than the number of time steps and random gaussians into the  DIFF_DECODER for spectrogram denoising
            #so all of this input is condensed into a tensor of size 256 by length_edited+original_number_of_mel_bins=edited_mel2ph.size(1). Actually such a tensor can hold more than all of the information contained by the inputs, as seen from the following:

            #print('Edited Text Tokens Size:')
            #print(edited_txt_tokens.size())
            #print('time_mel_masks Size:')
            #print(time_mel_masks.size())
            #print('edited_mel2ph Size:')
            #print(edited_mel2ph.size())
            #print('spk_embed Size:')
            #print(sample['spk_embed'].size())
            #print('ref_mels Size:')
            #print(ref_mels.size())
            #print('edited_f0 Size:')
            #print(edited_f0.size())
            #print('edited_uv Size:')
            #print(edited_uv.size())

            #Z: it is not clear at all what energy does

            output = self.model(edited_txt_tokens, time_mel_masks=time_mel_masks, mel2ph=edited_mel2ph, spk_embed=sample['spk_embed'],spk_id=sample['spk_ids'],
                       ref_mels=ref_mels, f0=edited_f0, uv=edited_uv,  infer=True, use_pred_pitch=True) 
            #start_vocoder_time=time.time()
            #print(f'Total Get Output Time {start_vocoder_time-start_get_output_time}')
            #passes to forward in GaussianDiffusion
            mel_out = output['mel_out'] * time_mel_masks + ref_mels * (1-time_mel_masks)
            wav_out = self.run_vocoder(mel_out)
            wav_gt = self.run_vocoder(sample['mel'])
            # item_name = sample['item_name'][0]
            # np.save(f'inference_acl/mel2ph/{item_name}',output['mel2ph'].cpu().numpy()[0])
        #print(f'Vocoder Time {time.time()-start_vocoder_time}')
        wav_out = wav_out.cpu().numpy()
        wav_gt = wav_gt.cpu().numpy()
        mel_out = mel_out.cpu().numpy()
        mel_gt = sample['mel'].cpu().numpy()
        masked_mel_out = ref_mels.cpu().numpy()
        masked_mel_gt = (sample['mel'] * time_mel_masks_orig[:, :, None]).cpu().numpy()

        return wav_out[0], wav_gt[0], mel_out[0], mel_gt[0], masked_mel_out[0], masked_mel_gt[0]

    
    #use input and MFA textgrid from data_preprocess if use_MFA = True or alignment from whisperX if use_MFA=False to get all inputs needed for a forward pass of the model  
    def preprocess_input(self, inp,use_MFA):
        """

        :param inp: {'item_name': (str,optional), 'item_name': (str,optional), 'text': str,'edited_text':str, 'words_region': '[int,int]', 'edited_words_region': '[int,int]','mfa_textgrid': str, 'mel':numpy array of generated mel spectrogram, 'wav': numpy array of loaded audio file}

        :return: {'item_name': (str,optional), 'text': str, 'ph': str of CMU phonemes corresponding to text,
                'ph2word': list of integers describing which word each phoneme in ph corresponds to, 'edited_ph2word': edited_ph2word,
                'ph_token': list of integers corresponding to encoding of ph, 'edited_ph_token': edited_ph_token,
                'words_region': [int,int] corresponding to word region (edited to include silences as words), 'edited_words_region': edited_words_region,
                'mel2ph': phoneme number being uttered over time, 'mel2word': mel2word, 
                'dur': time duration in bins of each phoneme,
                'f0': log fundamental pitch over time, 
                'uv': unvoiced (True) or voiced (False) over time,
                'mel': mel, 'wav': wav}
        """
        
        preprocessor = self.preprocessor
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')

        # Get phonemes for original txt using enG2p, then encode those phonemes to integers according to where they appear in the dictionaries located at binary_data_directory

        #txt is of the form 'this is an example sentence .'
        # words is of the form '<BOS> this | is | an | example | sentence <EOS>' 
        #ph is of the form  '<BOS> DH IH1 S | IH1 Z | AE1 N | IH0 G Z AE1 M P AH0 L | S EH1 N T AH0 N S <EOS>'
        # ph2word is of the form [1,2,2,2,3,4,4,5,6,6,7,8,8,8,8,8,8,8,8,9,10,10,10,10,10,10,10,11]
        ph, txt, words, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw)
        ph_token = self.ph_encoder.encode(ph)
       
        
        edited_text_raw = inp['edited_text']
        edited_ph, _, edited_words, edited_ph2word, _ = preprocessor.txt_to_ph(
            preprocessor.txt_processor, edited_text_raw)
        # Get phonemes for edited txt '' '' 
        edited_ph_token = self.ph_encoder.encode(edited_ph)

        #print('')
        #print('ph')
        #print(ph)
        #print('')
        #print('ph length')
        #print(len(ph.split(' ')))
        #print('')
        #print('edited_ph')
        #print(edited_ph)
        #print('')
        #print('')
        #print('edited_ph length')
        #print(len(edited_ph.split(' ')))
        #print('')

        # Get words_region
        words = words.split(' ')
        edited_words = edited_words.split(' ')

        #convert the strings representing the word region to ints and get word regions which are adjusted for silent phonemes. 
        # For instance, if the original audio had transcription "this is an example sentence" and the user input to [2,3] to replace is and an, the edited words region would be [4,6] (since we count <BOS> and spaces)
        region, edited_region = parse_region_list_from_str(inp['region']), parse_region_list_from_str(
            inp['edited_region'])
        words_region = get_words_region_from_origintxt_region(words, region)
        edited_words_region = get_words_region_from_origintxt_region(edited_words, edited_region)
        

        # Generate forced alignment either using the MFA textgrid from data_preprocess if use_MFA = True or alignment from whisperX if use_MFA=False 
        #what is actually used is mel2ph and dur. mel2ph is an array of integers like [1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3,3...] of length equal to the first dimension of mel (from the original audio)
        # which indicates which phoneme in ph is being uttered during each time window in the mel spectrogram 
        #dur is a list of integers of length equal to the number of phonemes (assuming a silent phoneme in between every phoneme change) which encodes for how many mel bins each phoneme is held for in the original audio
        #the numbers in mel2ph should correspond to which phoneme is being uttered in ph_list = ph.split(' ') indexed from 1
        wav = inp['wav']
        mel = inp['mel']
        mfa_textgrid = inp['mfa_textgrid']
        #get_align_from_mfa_output_time_start=time.time()
        #print(ph_token)
        mel2ph, dur,sil_frames = get_align_from_mfa_output(mfa_textgrid, ph, ph_token, mel,wav,use_MFA,self.whisper_processor,self.whisper_align_model,self.device)

        #print('mel2ph change times:')
        #find_what_times_mel2ph_changes(mel2ph)
        #print(f'get_align_from_mfa_output total time ={time.time()-get_align_from_mfa_output_time_start} - this is all done in CPU except for whisperX if enabled')

        #same as mel2ph but for words
        mel2word = [ph2word[p - 1] for p in mel2ph]  # [T_mel]

        #print('mel2word change times')
        #find_what_times_mel2ph_changes(mel2word)

        # Extract frame-level f0 and uv (pitch info). Uses parselmouth.Sound(wav_data, audio_sample_rate).to_pitch_ac(...) to get a numerical representation of the fundamental frequency of the original audio for each mel bin, then takes log base 2
        # f0 is equal in length to the first dim of mel
        # uv (unvoiced) is of the same length and only contains 0 or 1, with 1 indicating there was no utterance during that mel bin and 0 indicating there was an utterance during that mel bin
        f0, uv = extract_f0_uv(wav, mel)

        if inp['spk_id'] is not None:
            speaker_id=torch.LongTensor([0])
        else: 
            speaker_id=None


        item = {'item_name': item_name, 'text': txt, 'ph': ph,
                'ph2word': ph2word, 'edited_ph2word': edited_ph2word,
                'ph_token': ph_token, 'edited_ph_token': edited_ph_token,
                'words_region': words_region, 'edited_words_region': edited_words_region,
                'mel2ph': mel2ph, 'mel2word': mel2word, 'dur': dur,
                'f0': f0, 'uv': uv,
                'mel': mel, 'wav': wav, 'spk_id':speaker_id}
        return item,sil_frames

    #convert output of preprocess_input to tensors for input into diffusion model. Also gets speaker embedding using the original loaded .wav file
    #spk_embed is a tensor of size [1,256] containing floats
    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        edited_ph2word = torch.LongTensor(item['edited_ph2word'])[None, :].to(self.device)
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        dur = torch.LongTensor(item['dur'])[None, :].to(self.device)
        mel2word = torch.LongTensor(item['mel2word'])[None, :].to(self.device)
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        edited_txt_tokens = torch.LongTensor(item['edited_ph_token'])[None, :].to(self.device)
        if item['spk_id'] is not None:
            spk_ids = item['spk_id'].to(self.device)
        else:
            spk_ids=item['spk_id']

        # masked prediction related
        mel = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        wav = torch.FloatTensor(item['wav'])[None, :].to(self.device)

        # get spk embed using resemblyzer model
        #at this point the wav file is again converted into a spectrogram with librosa, but with different parameters and in chunks
        spk_embed = self.spk_embeding.embed_utterance(item['wav'].astype(float))
        spk_embed = torch.FloatTensor(spk_embed[None, :]).to(self.device)

        # get frame-level f0 and uv (pitch info)
        f0 = torch.FloatTensor(item['f0'])[None, :].to(self.device)
        uv = torch.FloatTensor(item['uv'])[None, :].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'ph2word': ph2word,
            'edited_ph2word': edited_ph2word,
            'mel2ph': mel2ph,
            'mel2word': mel2word,
            'dur': dur,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'edited_txt_tokens': edited_txt_tokens,
            'words_region': item['words_region'],
            'edited_words_region': item['edited_words_region'],
            'spk_ids': spk_ids,
            'mel': mel,
            'wav': wav,
            'spk_embed': spk_embed,
            'f0': f0,
            'uv': uv
        }
        return batch

  

    #runs inference on each dictionary given as input to the list dataset_info
    def example_run(self, dataset_info,use_MFA,use_librosa,save_wav_bool,disp_wav,mask_loc_buffer,inference_output_dir=None,speaker_id=None,fix_silent_phonemes=True):
        result_wavs=[]
        if type(mask_loc_buffer)!=int:
            print('Argument mask_loc_buffer must be a positive integer')
            return 1
        if mask_loc_buffer<0:
            print('Argument mask_loc_buffer must be a positive integer')
            return 1
    
            

        #perform inference editing one audio file
        def infer_one(data_info):
            global inference_output_dir
            #wav2spec_time_start=time.time()

            #their implementation for generating loading the .wav file and generating the mel spectrogram using librosa
            if use_librosa:
                wav2spec_res = librosa_wav2spec(data_info['wav_fn_orig'], fmin=55, fmax=7600, sample_rate=22050)
                #print(f'Generate Spectrogram Time: {time.time()-wav2spec_time_start} - you might consider improving performance by using torchaudio here instead of librosa by setting use_librosa=False')
                inp = {
                'item_name': data_info['item_name'],
                'text': data_info['text'],
                'edited_text': data_info['edited_text'],
                'region': data_info['region'],
                'edited_region': data_info['edited_region'],
                'mfa_textgrid': data_info['mfa_textgrid'],
                'mel': wav2spec_res['mel'],
                'wav': wav2spec_res['wav']
                }

            #if use_librosa=False, load the audio and generate the mel spectrogram with torchaudio instead. This is just recreating their librosa_wav2spec in torch
            else:
                #it seems we should put the wav directly on cuda but this causes errors down the road
                #eps=torch.tensor(1e-6,device=device)
                eps=torch.tensor(1e-6)
                wav,rate = torchaudio.load(data_info['wav_fn_orig'])
                wav = torchaudio.functional.resample(wav, orig_freq=rate, new_freq=22050)[0].squeeze()#.to(device)
                #loading in the wav file. A tensor of numbers representing the wav form over time of length 22050*(length of file in seconds)

                audio_to_mel = torchaudio.transforms.Spectrogram(
                        hop_length=256,
                        win_length=1024,
                        n_fft=1024,
                        power=1,
                        normalized=False,
                        pad_mode="constant"
                    )#.to(device)
                
                mel_scale = torchaudio.transforms.MelScale(
                        sample_rate=22050,
                        n_stft=1024 // 2 + 1,
                        n_mels=80,
                        f_min=55,
                        f_max=7600,
                        norm="slaney",
                        mel_scale="slaney",
                    )#.to(device)
                
                spec = audio_to_mel(wav)
                mel = mel_scale(spec)
                mel = torch.log10(torch.maximum(eps, mel)).transpose(0,1)  
                #mel is the mel spectrogram, shape is roughly [int(22050*(length of file in seconds)/256),80], with the value at [i,j] corresponding to the volume intensity of the jth pitch bin during ith time bin (roughly i*256/22050 seconds into the audio) 

                #pad the loaded loaded file with zeros at the end to make sure that its length divides into the hop size of 256 in the mel spectrogram
                pad = (wav.shape[0] // 256 + 1) * 256 - wav.shape[0]
                wav = torch.nn.functional.pad(wav, (0, pad), mode='constant', value=0.0)
                wav = wav[:mel.shape[0] * 256]
                

                #print(f'Generate Spectrogram Time: {time.time()-wav2spec_time_start}')
                inp = {
                'item_name': data_info['item_name'],
                'text': data_info['text'],
                'edited_text': data_info['edited_text'],
                'region': data_info['region'],
                'edited_region': data_info['edited_region'],
                'mfa_textgrid': data_info['mfa_textgrid'],
                'mel': mel.numpy(),
                'wav': wav.numpy(),
                'spk_id':speaker_id
                }

            #pass the provided input text, edited text, region, edited region, generated mel spectrogram, and loaded .wav file to infer_once for inference
            wav_out, wav_gt, mel_out, mel_gt, masked_mel_out, masked_mel_gt = self.infer_once(inp,use_MFA,mask_loc_buffer,fix_silent_phonemes)
            if save_wav_bool:
                #save_wav_time_start=time.time()

                #wav_out is the new wav generated from inference. wav_gt is the result of generating a mel spectrogram from the original audio and running it through the vocoder, and should be essentially identical to the source .wav
                save_wav(wav_out, f'{inference_output_dir}/{inp["item_name"]}_modified.wav', hp['audio_sample_rate'])
                save_wav(wav_gt, f'{inference_output_dir}/{inp["item_name"]}_ref.wav', hp['audio_sample_rate'])

                #print(f'Save wav time = {time.time()-save_wav_time_start}')
            return wav_out,wav_gt
        
        #make the inference output directory if it does not exist and then perform inference on each item in the list
        #ToDo: Batch inference?
        if save_wav_bool:
            os.makedirs(inference_output_dir, exist_ok=True)
        for item in dataset_info:
            wav_out,wav_gt=infer_one(item)

            if disp_wav:
                #ipd_disp_time_start=time.time()
                
                print('Ground Truth audio:')
                ipd.display(ipd.Audio(data = wav_gt, rate = 22050, autoplay = False))
                print('Inferred audio:')
                ipd.display(ipd.Audio(data = wav_out, rate = 22050, autoplay = False))

                #print(f'Ipd Display Time: {time.time()-ipd_disp_time_start}')
            result_wavs.append([wav_gt,wav_out])
        return result_wavs
    
    def infer_once(self, inp,use_MFA,mask_loc_buffer,fix_silent_phonemes):
        #preproc_input_start_time=time.time()
        inp,sil_frames = self.preprocess_input(inp,use_MFA)
        #forward_total_start_time=time.time()
        #print(f'Preprocess Input Time: {forward_total_start_time-preproc_input_start_time}')
        output = self.forward_model(inp,mask_loc_buffer,sil_frames,fix_silent_phonemes)
        #print(f'Forward Pass Total Time (includes everything after Preprocess Input Time): {time.time()-forward_total_start_time}')
        
        return output


    def build_vocoder(self):
        build_vocoder_start_time=time.time()
        base_dir = self.hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        config = set_hparams(config_path, global_hparams=True)
        vocoder = HifiGanGenerator(config)
        load_ckpt(vocoder, base_dir, 'model_gen')
        print(f'Build Vocoder Time {time.time()-build_vocoder_start_time}')
        print(f'Vocoder Device {self.device}')
        return vocoder


    def run_vocoder(self, c):
        c = c.transpose(2, 1)
        #voc_time_again=time.time()
        y = self.vocoder(c)[:, 0]
        #print(f'Voc inf time {time.time()-voc_time_again}')
        return y
    
    def test_phoneme_enoder(self,text):
        ph, _, _, _, _ = self.preprocessor.txt_to_ph(self.preprocessor.txt_processor, text)
        return [self.ph_encoder.encode(ph),ph]


#for each row in the .csv passed to  file_path in data_preprocess, add a dictionary to a list containing the column info in that .csv

def load_dataset_info(file_path):
    dataset_frame = pd.read_csv(file_path)
    dataset_info = []
    for index, row in dataset_frame.iterrows():
        row_info = {}
        row_info['item_name'] = row['item_name']
        row_info['text'] = row['text']
        row_info['edited_text'] = row['edited_text']
        row_info['wav_fn_orig'] = row['wav_fn_orig']
        row_info['edited_region'] = row['edited_region']
        row_info['region'] = row['region']
        dataset_info.append(row_info)
    return dataset_info


# preprocess data with MFA forced alignment if use_MFA=true. Otherwise the entry with key 'mfa_textgrid' in the dataset_info dictionary is unused, so this just returns the same thing as load_dataset_info

def data_preprocess(file_path, input_directory, dictionary_path, acoustic_model_path, output_directory, align=True):

    assert os.path.exists(file_path) and os.path.exists(input_directory) and os.path.exists(acoustic_model_path), \
        f"{file_path},{input_directory},{dictionary_path},{acoustic_model_path}"
    dataset_info = load_dataset_info(file_path)
    for data_info in dataset_info:
        data_info['mfa_textgrid'] = f'{output_directory}/{data_info["item_name"]}.TextGrid'
    if not align:
        return dataset_info

    from data_gen.tts.txt_processors.en import TxtProcessor
    txt_processor = TxtProcessor()

    # gen  .lab file. Uses a g2p model from g2p_en and nltk to convert the input sentence to the CMU Pronouncing Dictionary and add the words to the MFA dict at dictionary_path if they werent there already 
    
    def gen_forced_alignment_info(data_info):
        *_, ph_gb_word = BasePreprocessor.txt_to_ph(txt_processor, data_info['text'])
        tg_fn = f'{input_directory}/{data_info["item_name"]}.lab'
        ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                     for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
        with open(tg_fn, 'w') as f_txt:
            f_txt.write(ph_gb_word_nosil)

        with open(dictionary_path, 'r') as f:  # update mfa dict for unseen word
            lines = f.readlines()

        with open(dictionary_path, 'a+') as f:
            for item in ph_gb_word_nosil.split(" "):
                item = item + '\t' + ' '.join(item.split('_')) + '\n'
                if item not in lines:
                    f.writelines([item])

    for item in dataset_info:
        gen_forced_alignment_info(item)
        item_txt,item_name, wav_fn_orig = item['text'],item['item_name'], item['wav_fn_orig']                      #there were some typos in the original script. Not even sure if this is necessary
        shutil.copyfile(wav_fn_orig,f'inference/audio/{item_name}.wav')
        with open(f'inference/audio/{item_name}.txt', 'w') as file:
            file.write(item_txt)
        #os.system(f'cp -f {wav_fn_orig} inference/audio/{item_name}.wav')

    print("Generating forced alignments with mfa. Please wait for about several minutes.")
    mfa_out = output_directory
    if os.path.exists(mfa_out):
        shutil.rmtree(mfa_out)
    
    #if you really want to use MFA for your phoneme alignment and are having trouble getting consistent output, you might try uncommenting the below

    #for item in dataset_info:
    #    gen_forced_alignment_info(item)
    #    item_name = item['item_name']                    
    #    command = ' '.join(
    #        ['mfa align_one -j 4 --clean', f'inference/audio/{item_name}.wav',f'inference/audio/{item_name}.txt', dictionary_path, acoustic_model_path, output_directory+'/'+str(item_name)+'.TextGrid','--use_mp','--single_speaker'])
    #    #changed to align_one
    #    print(command)
    #    os.system(command)

    command = ' '.join(
            ['mfa align -j 4 --clean', f'inference/audio', dictionary_path, acoustic_model_path, output_directory,'--use_mp','--single_speaker']) #adding '--beam 100' can make this work better
    print(command)
    os.system(command)
    return dataset_info 

def find_what_times_mel2ph_changes(mel2ph):
    mel2ph_time_changes=[]
    for i in range(1,len(mel2ph)):
        last_mel_val=mel2ph[i-1]
        cur_mel_val=mel2ph[i]
        if last_mel_val!=cur_mel_val:
            mel2ph_time_changes.append(f'{last_mel_val} to {cur_mel_val} : {i*256/22050}')
    print(mel2ph_time_changes)

def get_silence_regions_from_uv(uv):
    sil_reg_from_uv=[]
    sil_beg=0
    for i in range(len(uv)):
        last_uv=uv[i-1]
        cur_uv=uv[i]
        if last_uv!=cur_uv:
            if last_uv==0:
                sil_beg=i
            if last_uv==1:
                sil_end=i-1
                sil_reg_from_uv.append([sil_beg,sil_end])
        if (i==len(uv)-1)and(cur_uv==1):
            sil_reg_from_uv.append([sil_beg,i])
    return(sil_reg_from_uv)





if __name__ == '__main__':
    # you can use 'align' to choose whether using MFA during preprocessing
    total_time_start=time.time()
    test_file_path = 'inference/example2.csv'
    test_wav_directory = 'inference/audio'
    dictionary_path = 'data/processed/libritts/mfa_dict.txt'
    acoustic_model_path = 'data/processed/libritts/mfa_model.zip'
    output_directory = 'inference/audio/mfa_out'
    use_MFA=False
    use_librosa=False 
    #os.system('rm -r inference/audio')
    os.makedirs(f'inference/audio', exist_ok=True)
    if use_MFA:
        dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
                                   output_directory, align=True)
    else: 
        dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
                                   output_directory, align=False)
    
    hparams=set_hparams(exp_name='spec_denoiser')

    Espeak_dll_directory = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
    EspeakWrapper.set_library(Espeak_dll_directory) 

    binary_data_directory='C:\\Users\\bezem\\Documents\\erdos_deep_learning\\Speech-Editing-Toolkit-stable\\data\\processed\\binary\\libritts'
    whisperX_model_directory='C:\\Users\\bezem\\Documents\\erdos_deep_learning\\whisperX-main\\facebook'
    mfa_dictionary_path = 'data/processed/libritts/mfa_dict.txt'
    mfa_acoustic_model_path = 'data/processed/libritts/mfa_model.zip'
    
    infer_class_obj=SpecDenoiserInfer(hparams,binary_data_directory,whisperX_model_directory)    
    infer_class_obj.example_run(dataset_info,use_MFA,use_librosa)
    print(f'Total Time: {time.time()-total_time_start}')