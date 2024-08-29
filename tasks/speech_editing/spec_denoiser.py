import torch

import utils
from utils.commons.hparams import hparams
from utils.audio.pitch.utils import denorm_f0
from modules.speech_editing.spec_denoiser.spec_denoiser import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from tasks.speech_editing.speech_editing_base import SpeechEditingBaseTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from tasks.speech_editing.dataset_utils import StutterSpeechDataset

from inference.tts.infer_utils import get_align_from_mfa_output
from transformers import AutoProcessor, AutoModelForCTC
import logging
import time
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from data_gen.tts.base_preprocess import BasePreprocessor
import torchaudio

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class SpeechDenoiserTask(SpeechEditingBaseTask):
    def __init__(self):
        super(SpeechDenoiserTask, self).__init__()
        self.dataset_cls = StutterSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()

        #Espeak_dll_directory = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
        #EspeakWrapper.set_library(Espeak_dll_directory) 
        #whisperX_model_directory='C:\\Users\\bezem\\Documents\\erdos_deep_learning\\whisperX-main\\facebook'
        #self.whisper_processor,self.whisper_align_model = self.build_whisper(whisperX_model_directory)
        #self.whisper_align_model.eval()
        #self.whisper_align_model.to('cuda')
        #self.preprocessor= BasePreprocessor()
#
    #def build_whisper(self,whisperX_model_directory):
        #logging.getLogger('transformers').setLevel(logging.ERROR)
        #whisperX_load_start_time=time.time()
        #processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft",cache_dir=whisperX_model_directory)
        #align_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft",cache_dir=whisperX_model_directory)
        #print(f'WHISPERX LOAD TIME = {time.time()-whisperX_load_start_time}')
#
#        
        #return processor, align_model
        if hparams['single_speaker_ada_fine_tune']:
            self.fine_tuning_param_names=self.get_single_speaker_ada_fine_tune_list()
        elif hparams['naive_fine_tune']:
            self.fine_tuning_param_names='fs.spk_id_proj.weight'


    def get_single_speaker_ada_fine_tune_list(self):

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
        if hparams['fine_tune_weight_sel']=='':
            all_weights_to_fine_tune_names=encoder_layer_norms+decoder_layer_norms+dur_predictor_layer_norms+pitch_predictor_layer_norms+speaker_embedding_weights
        else:
            selection =list(hparams['fine_tune_weight_sel'])
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
        if hparams['use_spk_id']:
            all_weights_to_fine_tune_names=all_weights_to_fine_tune_names+['fs.spk_id_proj.weight']
        print(f'Fine tuning {len(all_weights_to_fine_tune_names)} named parameters')
        return all_weights_to_fine_tune_names

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = GaussianDiffusion(
            phone_encoder=self.token_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )


    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']

        #mfa_textgrid=''
        #ph, _, _, _, _ = self.preprocessor.txt_to_ph(self.preprocessor.txt_processor, sample['text'][0])
#        
        #eps=torch.tensor(1e-6)
        #wav,rate = torchaudio.load(sample['wav_fn'][0])
        #wav = torchaudio.functional.resample(wav, orig_freq=rate, new_freq=22050)[0].squeeze()#.to(device)
        ##loading in the wav file. A tensor of numbers representing the wav form over time of length 22050*(length of file in seconds)
#
        #audio_to_mel = torchaudio.transforms.Spectrogram(
                        #hop_length=256,
                        #win_length=1024,
                        #n_fft=1024,
                        #power=1,
                        #normalized=False,
                        #pad_mode="constant"
                    #)#.to(device)
#                
        #mel_scale = torchaudio.transforms.MelScale(
                        #sample_rate=22050,
                        #n_stft=1024 // 2 + 1,
                        #n_mels=80,
                        #f_min=55,
                        #f_max=7600,
                        #norm="slaney",
                        #mel_scale="slaney",
                    #)#.to(device)
#                
        #spec = audio_to_mel(wav)
        #mel = mel_scale(spec)
        #mel = torch.log10(torch.maximum(eps, mel)).transpose(0,1)  
                ##mel is the mel spectrogram, shape is roughly [int(22050*(length of file in seconds)/256),80], with the value at [i,j] corresponding to the volume intensity of the jth pitch bin during ith time bin (roughly i*256/22050 seconds into the audio) 
#
                ##pad the loaded loaded file with zeros at the end to make sure that its length divides into the hop size of 256 in the mel spectrogram
        #pad = (wav.shape[0] // 256 + 1) * 256 - wav.shape[0]
        #wav = torch.nn.functional.pad(wav, (0, pad), mode='constant', value=0.0)
        #wav = wav[:mel.shape[0] * 256]
        ##print(sample['txt_tokens'][0].to('cpu').numpy())
        ##print(len(sample['txt_tokens'][0].to('cpu').numpy()))
        #mel2ph,_,_=get_align_from_mfa_output(mfa_textgrid, ph, sample['txt_tokens'][0].to('cpu').numpy(), sample['mels'][0].to('cpu').numpy(),wav,use_MFA=False,processor=self.whisper_processor,align_model=self.whisper_align_model,device='cuda')
#
        #mel2ph = torch.LongTensor(mel2ph)[None, :].to('cpu')

        f0 = sample['f0']
        uv = sample['uv']
        time_mel_masks = sample['time_mel_masks'][:,:,None]
        spk_embed = sample.get('spk_embed') #if not hparams['use_spk_id'] else sample.get('spk_ids')
        spk_id=sample.get('spk_ids')
        output = self.model(txt_tokens, time_mel_masks, mel2ph=mel2ph, spk_embed=spk_embed,spk_id=spk_id,
                       ref_mels=target, f0=f0, uv=uv, infer=infer)

        losses = {}
        self.add_mel_loss(output['mel_out']*time_mel_masks, target*time_mel_masks, losses, postfix="_coarse")
        output['mel_out'] = output['mel_out']*time_mel_masks + target*(1-time_mel_masks)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        # if hparams['use_energy_embed']:
        #     self.add_energy_loss(output['energy_pred'], energy, losses)
        if not infer:
            return losses, output
        else:
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']

        energy = None
        spk_embed = sample.get('spk_embed') #if not hparams['use_spk_id'] else 
        spk_id=sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        time_mel_masks = sample['time_mel_masks'][:,:,None]

        outputs['losses'] = {}
        outputs['losses'], output = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, time_mel_masks, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=target,spk_id=spk_id, infer=True)
            model_out['mel_out'] = model_out['mel_out']*time_mel_masks + target*(1-time_mel_masks)
            # gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=None, f0=None)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'])
        return outputs

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = None
        f0 = None
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
