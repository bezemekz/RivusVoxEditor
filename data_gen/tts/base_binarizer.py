import json
import os
import sys
sys.path.append(os.getcwd())
import random
import traceback
from functools import partial

import numpy as np
from resemblyzer import VoiceEncoder
from tqdm import tqdm

import utils.commons.single_thread_env  # NOQA
from utils.audio import librosa_wav2spec
from utils.audio.align import get_mel2ph, mel2token_to_dur
from utils.audio.cwt import get_lf0_cwt, get_cont_lf0
from utils.audio.pitch.utils import f0_to_coarse
from utils.audio.pitch_extractors import extract_pitch
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import remove_file, copy_file

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
import logging
import time
from phonemizer.backend.espeak.wrapper import EspeakWrapper

np.seterr(divide='ignore', invalid='ignore')


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self):
        self.dataset_name = 'NarratorBG3'
        self.processed_data_dir = f'data/processed/{self.dataset_name}'
        self.binary_data_dir = f'data/binary/{self.dataset_name}'
        self.items = {}
        self.item_names = []
        self.shuffle = True
        self.with_spk_embed = True
        self.with_wav = False

        # text2mel parameters
        self.text2mel_params = {'fft_size': 1024, 'hop_size': 256, 'win_size': 1024,
                                'audio_num_mel_bins': 80, 'fmin': 55, 'fmax': 7600,
                                'f0_min': 80, 'f0_max': 600, 'pitch_extractor': 'parselmouth',
                                'audio_sample_rate': 22050, 'loud_norm': False,
                                'mfa_min_sil_duration': 0.1, 'trim_eos_bos': False,
                                'with_align': True, 'text2mel_params': False,
                                'dataset_name': self.dataset_name,
                                'with_f0': True, 'min_mel_length': 64}
        
        Espeak_dll_directory = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
        EspeakWrapper.set_library(Espeak_dll_directory) 
        whisperX_model_directory='C:\\Users\\bezem\\Documents\\erdos_deep_learning\\whisperX-main\\facebook'
        self.whisper_processor,self.whisper_align_model = self.build_whisper(whisperX_model_directory)
        self.whisper_align_model.eval()
        self.whisper_align_model.to('cuda')

        self.data_size=len(os.listdir(self.processed_data_dir+'/wav_processed'))
        self.val_percent=.2

    def build_whisper(self,whisperX_model_directory):
        logging.getLogger('transformers').setLevel(logging.ERROR)
        whisperX_load_start_time=time.time()
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft",cache_dir=whisperX_model_directory)
        align_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft",cache_dir=whisperX_model_directory)
        print(f'WHISPERX LOAD TIME = {time.time()-whisperX_load_start_time}')

        
        return processor, align_model
        


    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.shuffle:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        # 400 for seen, 4182 for unseen
        start=int(np.ceil(self.data_size*self.val_percent)+1)
        return self.item_names[start:]

    @property
    def valid_item_names(self):
        end = int(np.floor(self.data_size*self.val_percent))
        return self.item_names[:end]

    @property
    def test_item_names(self):
        end = int(np.floor(self.data_size*self.val_percent))
        return self.item_names[:end]

    def _convert_range(self, range_):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(self.binary_data_dir, exist_ok=True)
        for fn in ['phone_set.json', 'word_set.json', 'spk_map.json']:
            remove_file(f"{self.binary_data_dir}/{fn}")
            copy_file(f"{self.processed_data_dir}/{fn}", f"{self.binary_data_dir}/{fn}")
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = self.binary_data_dir
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item)
        ph_lengths = []
        mel_lengths = []
        total_sec = 0
        items = []
        #args = [{'item': item, 'text2mel_params': self.text2mel_params} for item in meta_data]
        #for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
        for data in tqdm(meta_data, desc='Processing data'):
            item = BaseBinarizer.process_item(data,self.text2mel_params,self.whisper_processor,self.whisper_align_model)
            if item is not None:
                items.append(item)
        if self.with_spk_embed:
            ve=VoiceEncoder().cuda()
            wavs = [item['wav'] for item in items]
            for item_id,wav in tqdm(enumerate(wavs),desc='Extracting spk embed'):
                spk_embed=self.get_spk_embed(wav,{'voice_encoder':ve})
                items[item_id]['spk_embed'] = spk_embed
                if spk_embed is None:
                    del items[item_id]
            #for item_id, spk_embed in multiprocess_run_tqdm(
            #        self.get_spk_embed, args,
            #        init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().cuda()}, num_workers=2,
            #        ):


        for item in items:
            if not self.with_wav and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            mel_lengths.append(item['len'])
            assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, text2mel_params,whisper_processor,whisper_align_model):
        item['ph_len'] = len(item['ph_token'])
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, text2mel_params)
        if len(mel) < text2mel_params['min_mel_length']:
            return None
        try:
            # stutter label
            # cls.process_stutter_label(wav, mel, item, text2mel_params)
            # alignments
            n_bos_frames, n_eos_frames = 0, 0
            if text2mel_params['with_align']:
                tg_fn = f"data/processed/{text2mel_params['dataset_name']}/mfa_outputs/{item_name}.TextGrid"
                item['tg_fn'] = tg_fn
                cls.process_align(tg_fn, item, text2mel_params,whisper_processor,whisper_align_model)
                if text2mel_params['trim_eos_bos']:
                    n_bos_frames = item['dur'][0]
                    n_eos_frames = item['dur'][-1]
                    T = len(mel)
                    item['mel'] = mel[n_bos_frames:T - n_eos_frames]
                    item['mel2ph'] = item['mel2ph'][n_bos_frames:T - n_eos_frames]
                    item['mel2word'] = item['mel2word'][n_bos_frames:T - n_eos_frames]
                    item['dur'] = item['dur'][1:-1]
                    item['dur_word'] = item['dur_word'][1:-1]
                    item['len'] = item['mel'].shape[0]
                    item['wav'] = wav[n_bos_frames * text2mel_params['hop_size']:len(wav) - n_eos_frames * text2mel_params['hop_size']]
            if text2mel_params['with_f0']:
                cls.process_pitch(item, n_bos_frames, n_eos_frames, text2mel_params)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

    @classmethod
    def process_audio(cls, wav_fn, res, text2mel_params):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=text2mel_params['fft_size'],
            hop_size=text2mel_params['hop_size'],
            win_length=text2mel_params['win_size'],
            num_mels=text2mel_params['audio_num_mel_bins'],
            fmin=text2mel_params['fmin'],
            fmax=text2mel_params['fmax'],
            sample_rate=text2mel_params['audio_sample_rate'],
            loud_norm=text2mel_params['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        # if binarization_args['with_linear']:
        #     res['linear'] = wav2spec_dict['linear']
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / text2mel_params['audio_sample_rate'], 'len': mel.shape[0]})
        return wav, mel
    
    @classmethod
    def process_stutter_label(cls, wav, mel, res, text2mel_params):
        # obtain the stutter-oriented mel mask from stutter_label
        stutter_fn = f"data/processed/stutter_set/stutter_labels/{res['item_name'][:17]}/{res['item_name']}.npy"
        stutter_label = np.load(stutter_fn)
        stutter_mel_mask = np.zeros(mel.shape[0])
        if len(stutter_label) > 0:
            for item in stutter_label:
                stutter_start_time, stutter_end_time = item[0], item[1]
                stutter_start_frame = int(stutter_start_time * text2mel_params['audio_sample_rate'] // text2mel_params['hop_size'])
                stutter_end_frame = int(stutter_end_time * text2mel_params['audio_sample_rate'] // text2mel_params['hop_size'])
                if item[2] != 0:
                    item[2] = 1
                stutter_mel_mask[stutter_start_frame:stutter_end_frame] = item[2]
        res.update({'stutter_mel_mask': stutter_mel_mask})

    @staticmethod
    def process_align(tg_fn, item, text2mel_params,whisper_processor,whisper_align_model):
        ph = item['ph']
        mel = item['mel']
        ph_token = item['ph_token']

        eps=torch.tensor(1e-6)
        wav,rate = torchaudio.load(item['wav_fn'])
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

        if tg_fn is not None: #and os.path.exists(tg_fn):
            mel2ph, dur,sil_frames = get_mel2ph(tg_fn, ph, mel, text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],wav,False,
                                    whisper_processor,whisper_align_model,'cuda',text2mel_params['mfa_min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found for {tg_fn}")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}, for {tg_fn}")
        item['mel2ph'] = mel2ph
        item['dur'] = dur

        ph2word = item['ph2word']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, len(item['word_token']))
        item['dur_word'] = dur_word.tolist()  # [T_word]

    @staticmethod
    def process_pitch(item, n_bos_frames, n_eos_frames, text2mel_params):
        wav, mel = item['wav'], item['mel']
        f0 = extract_pitch(text2mel_params['pitch_extractor'], wav,
                         text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                         f0_min=text2mel_params['f0_min'], f0_max=text2mel_params['f0_max'])
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)
        item['f0'] = f0
        item['pitch'] = pitch_coarse
        # if hparams['binarization_args']['with_f0cwt']:
        #     uv, cont_lf0_lpf = get_cont_lf0(f0)
        #     logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        #     cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        #     cwt_spec, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        #     item['cwt_spec'] = cwt_spec
        #     item['cwt_mean'] = logf0s_mean_org
        #     item['cwt_std'] = logf0s_std_org

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))


if __name__ == '__main__':
    BaseBinarizer().process()
