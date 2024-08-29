from utils.text.text_encoder import is_sil_phoneme
import sys 
import whisperx
import time
from munch import Munch
import torchaudio
import librosa
import torch

def whisperX_TextGrid(ph,wav,audio_sample_rate,processor,align_model,device):

    pipeline_type='huggingface'
    
    audio_len = len(wav)/audio_sample_rate

    #resample to 16kHz for the wav2vec model
    wav=torchaudio.functional.resample(torch.tensor(wav), orig_freq=audio_sample_rate, new_freq=16000).to('cpu').numpy()
    
    align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}
    align_dictionary['|']=align_dictionary.pop('<pad>')
    metadata = {"language": 'en', "dictionary": align_dictionary, "type": pipeline_type}
    ipa_sentence, keep_track_IPA = cmu_ph_to_ipa_sentence(ph)

    phone_transcript = [{"text": ipa_sentence, "start":0, "end":audio_len}]

    #print(ph)
    #print(len(ph))
    #print(wav)
    #whisper_inference_time_start=time.time()
    with torch.no_grad():
        try:
            whisper_out = whisperx.align(transcript=phone_transcript, model=align_model, align_model_metadata=metadata, audio=wav, device=device, return_char_alignments=True,preprocess=False)
        except:
            whisper_out = whisperx.align(transcript=phone_transcript, model=align_model, align_model_metadata=metadata, audio=wav, device=device, return_char_alignments=True)


    
    #print(f'WHISPERX INFERENCE TIME: {time.time()-whisper_inference_time_start}')

    # NEED TO MANIPULATE WHISPER_OUT TO MAKE SURE THE IPA PHONEMES CAN BE RECONVERTED AS CMU PHONEMES.
    #print("Current whisper_out IPA chars: ")
    #for w in whisper_out['segments'][0]['chars']:
    #    print(w)
    
    whisper_out_chars = whisper_out['segments'][0]['chars']
    #print(len(whisper_out_chars))
    correctly_reconstructed_IPA_chars = reconstruct_correct_IPA_aligment(whisper_out_chars, keep_track_IPA)
    
    #print(len(correctly_reconstructed_IPA_chars))
    #print("Corrected whisper_out chars: ", correctly_reconstructed_IPA_chars)

    TextGrid=clean_translate_whisperx_ipa_char_result(correctly_reconstructed_IPA_chars)
    
    #append silence at end
    end_silence_itv=Munch()
    end_silence_itv.mark=''
    end_silence_itv.minTime=TextGrid[-1].maxTime
    end_silence_itv.maxTime=audio_len
    TextGrid.append(end_silence_itv)

    #print("TextGrid is of length : ", len(TextGrid))
    #for t in TextGrid:
    #    print(t)
    #print(len(TextGrid))

    return TextGrid


def reconstruct_correct_IPA_aligment(whisper_out_chars, keep_track_IPA):
    index_IPA = 0
    index_chars = 0
    new_dict_IPA_chars = []

    # Add initial " " character to the new_dict_IPA_chars
    new_dict_IPA_chars.append({"char": " "})

    while(index_chars in range(len(whisper_out_chars))):
        curr_IPA_ph_list = keep_track_IPA[index_IPA]
        #print("Curr IPA ph list and index: ", curr_IPA_ph_list, index_IPA)

        curr_char = whisper_out_chars[index_chars]['char']
        #print("Curr char and index: ", curr_char, index_chars)

        while curr_char == " ":
            #print("Going through silences")
            # If it is the beginning of the sentence, we don't have a start/end time, so ignore and move to the next index
            if index_chars == 0:
                #print("Initial silent phoneme; ignoring it.")
                index_chars += 1
                curr_char = whisper_out_chars[index_chars]['char']
                #print("Curr IPA ph list and index: ", curr_IPA_ph_list, index_IPA)
                #print("Curr char and index: ", curr_char, index_chars)
            else: 
                start_time = whisper_out_chars[index_chars]["start"]
                end_time = whisper_out_chars[index_chars]["end"]
                start_score = whisper_out_chars[index_chars]["score"]
                temp_dict = {"char": curr_char, "start": start_time, "end": end_time, "score": start_score}
                #print("Completed dict for this ph:", temp_dict)
                new_dict_IPA_chars.append(temp_dict)
                index_chars += 1
                curr_char = whisper_out_chars[index_chars]['char']
    
        
        if curr_char == curr_IPA_ph_list[0]:
            #print("We have a match between curr char and beginning of IPA_list")

            # Problem: if curr_char == "ˈ" or "ˌ" or , then we don't have an associated starting time. So we need to look at the starting time of the next character!
            if (curr_char == "ˈ") or (curr_char == "ˌ"):
                #print("Curr_char is ", curr_char)
                start_time = whisper_out_chars[index_chars+1]["start"]
                start_score = whisper_out_chars[index_chars+1]["score"]
            else:
                start_time = whisper_out_chars[index_chars]["start"]
                start_score = whisper_out_chars[index_chars]["score"]
            
            # End character index
            index_chars += len(curr_IPA_ph_list) - 1
            curr_char = whisper_out_chars[index_chars]["char"]
            
            # If out final character is ː , then ː has no "end" key, so we need to take the end_time from the preceding character
            if curr_char == "ː":
                #print("Curr_char is : :", curr_char)
                end_time = whisper_out_chars[index_chars-1]["end"]
                end_score = whisper_out_chars[index_chars-1]["score"]
            else:
                end_time = whisper_out_chars[index_chars]["end"]
                end_score = whisper_out_chars[index_chars]["score"]

            IPA_ph = "".join(curr_IPA_ph_list)
            temp_dict = {"char": IPA_ph, "start": start_time, "end": end_time, "score": (start_score + end_score)*0.5}
            #print("Completed dict for this ph:", temp_dict)
            new_dict_IPA_chars.append(temp_dict)
        index_chars += 1
        index_IPA += 1
    
    #for d in new_dict_IPA_chars:
    #    print(d)
    return new_dict_IPA_chars  


def cmu_ph_to_ipa_sentence(ph):
    #input is ph in inp which is input to get_mel2ph
    #output is the sentence in ipa form to input to whisperx.align 
    
    # We will need to keep track of the CMU2IPA translations of the phonemes
    # as some of them have more than one character, and we need to pass this info along.
    keep_track_IPA_chars = []
    words = ph.split('|')
    translated_sen=''
    for word in words:
        translated_word='' 
        phonemes=word.split(' ')
        for phoneme in phonemes: 
            if not is_sil_phoneme(phoneme):
                temp_transl = cmu_to_ipa_dict[phoneme]
                keep_track_IPA_chars.append([c for c in temp_transl])
                translated_word+=temp_transl
        if not translated_word=='':
            translated_sen+=" "
        translated_sen+=translated_word
    return (translated_sen), keep_track_IPA_chars

def clean_translate_whisperx_ipa_char_result(chars):
    #input is whisperx.align(phone_transcript, align_model, metadata, audio, device, return_char_alignments=True)['segments'][0]['chars']
    cleaned_result=[] #will hold a list of the same format returned by TextGrid.fromFile(file) where file is a textgrid output from MFA
    i=0 #index for the time stamped phonemes
    first_char_flag=True #to match MFA output we input silence at the beginning
    while i<len(chars):
        char_to_trans='' #will be the ipa character translated to mfa via the dictionary
        if chars[i]['char']==' ': #spaced are not given time stamps by whisperx so we skip them
            i+=1
        else:
            if (chars[i]['char']=='ˈ')or(chars[i]['char']=='ˌ'): #these characters are split up in the whisperx output. the timing for e.g. 'ɪ will be returned like {'char': 'ˈ'},{'char': 'ɪ', 'start': 0.089, 'end': 0.29, 'score': 0.886}, so we need to recombine to 'ɪ for translation
                char_to_trans+=chars[i]['char']
                i+=1 
            char_to_trans+=chars[i]['char']
            if first_char_flag:
                new_intvl=Munch()
                new_intvl.mark=''
                new_intvl.minTime=0.0
                new_intvl.maxTime=chars[i]['start']
                cleaned_result.append(new_intvl)
                #cleaned_result.append({'mark':'','minTime':0.0,'maxTime':chars[i]['start']}) #appending silence for the first character with time stamps
                first_char_flag=False
            if (i+1<len(chars)) and (chars[i+1]['char']=='ː'): #handling the same issue as for 'ˈ' and 'ˌ', except : comes after
                char_to_trans+='ː'
                if cleaned_result[-1]['maxTime']!=chars[i]['start']: #if the start time for the char we are about to append is greater than the endtime for the previous character, insert silence for that duration
                    new_intvl=Munch()
                    new_intvl.mark=''
                    new_intvl.minTime=cleaned_result[-1]['maxTime']
                    new_intvl.maxTime=chars[i]['start']
                    cleaned_result.append(new_intvl)
                    #cleaned_result.append({'mark':'','minTime':cleaned_result[-1]['maxTime'],'maxTime':chars[i]['start']})
                new_intvl=Munch()
                new_intvl.mark=ipa_to_cmu_dict[char_to_trans]
                new_intvl.minTime=chars[i]['start']
                new_intvl.maxTime=chars[i]['end']
                cleaned_result.append(new_intvl)
                #cleaned_result.append({'mark':ipa_to_cmu_dict[char_to_trans],'minTime':chars[i]['start'],'maxTime':chars[i]['end']})
                i+=2
            else:
                if cleaned_result[-1]['maxTime']!=chars[i]['start']: #if the start time for the char we are about to append is greater than the endtime for the previous character, insert silence for that duration
                    new_intvl=Munch()
                    new_intvl.mark=''
                    new_intvl.minTime=cleaned_result[-1]['maxTime']
                    new_intvl.maxTime=chars[i]['start']
                    cleaned_result.append(new_intvl)
                    #cleaned_result.append({'mark':'','minTime':cleaned_result[-1]['maxTime'],'maxTime':chars[i]['start']})
                new_intvl=Munch()
                new_intvl.mark=ipa_to_cmu_dict[char_to_trans]
                new_intvl.minTime=chars[i]['start']
                new_intvl.maxTime=chars[i]['end']
                cleaned_result.append(new_intvl)
                #cleaned_result.append({'mark':ipa_to_cmu_dict[char_to_trans],'minTime':chars[i]['start'],'maxTime':chars[i]['end']}) #translate using the dict, drop scores, and use the same keys as TextGrid.fromFile
                i+=1
                    

    return cleaned_result


##############
# CMU to IPA #
##############

cmu_to_ipa_dict={
    '<pad>': '<pad>',
    '<unk>': '<unk>',
    '<s>': '<s>',
    '</s>': '</s>',
    'AA0': 'ɑ',
    'AA1': 'ˈɑː',
    'AA2': 'ˌɑ',
    'AE0': 'æ',
    'AE1': 'ˈæ',
    'AE2': 'ˌæ',
    'AH0': 'ə',
    'AH1': 'ˈʌ',
    'AH2': 'ˌʌ',
    'AO0': 'ɔ',
    'AO1': 'ˈɔː',
    'AO2': 'ˌɔ',
    'AW0': 'aʊ',
    'AW1': 'ˈaʊ',
    'AW2': 'ˌaʊ',
    'AY0': 'aɪ',
    'AY1': 'ˈaɪ',
    'AY2': 'ˌaɪ',
    'B': 'b',
    'CH': 'tʃ',
    'D': 'd',
    'DH': 'ð',
    'EH0': 'ɛ',
    'EH1': 'ˈɛ',
    'EH2': 'ˌɛ',
    'ER0': 'ɚ',
    'ER1': 'ˈɚ',
    'ER2': 'ˌɚ',
    'EY0': 'eɪ',
    'EY1': 'ˈeɪ',
    'EY2': 'ˌeɪ',
    'F': 'f',
    'G': 'ɡː', #Z: edited this, 'g' is not in the dictionary for the ipa model but g: is
    'HH': 'h',
    'IH0': 'ɪ',
    'IH1': 'ˈɪ',
    'IH2': 'ˌɪ',
    'IY0': 'i',
    'IY1': 'ˈiː',
    'IY2': 'ˌi',
    'JH': 'dʒ',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'OW0': 'oʊ',
    'OW1': 'ˈoʊ',
    'OW2': 'ˌoʊ',
    'OY0': 'ɔɪ',
    'OY1': 'ˈɔɪ',
    'OY2': 'ˌɔɪ',
    'P': 'p',
    'R': 'ɹ',
    'S': 's',
    'SH': 'ʃ',
    'T': 't',
    'TH': 'θ',
    'UH0': 'ʊ',
    'UH1': 'ˈʊ',
    'UH2': 'ˌʊ',
    #'UW': 'uː',
    'UW0': 'u',
    'UW1': 'ˈuː',
    'UW2': 'ˌu',
    'V': 'v',
    'W': 'w',
    'Y': 'j',
    'Z': 'z',
    'ZH': 'ʒ'
}


ipa_to_cmu_dict = {value: key for key, value in cmu_to_ipa_dict.items()}
