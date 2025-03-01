import math
import os
import random as rd

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

def choose_random_segment(root, speaker_list,exclude = None):
    speaker = rd.choice(speaker_list)
    while exclude is not None and speaker == exclude:
        speaker = rd.choice(speaker_list)
    wav_folder = os.path.join(root,speaker)
    wav = rd.choice(os.listdir(wav_folder))
    return wav, speaker

def get_random_noise(noise_df):
    idx = rd.randint(0,len(noise_df)-1)
    return noise_df.iloc[idx]['utterance_id']

def prepare_aishell_mix_dataset(
    aishell_3_folder_path: str,
    wham_noise_folder_path: str,
    output_folder_path: str,
):
    rd.seed(5122006)
    aishell_train_dir = os.path.join(aishell_3_folder_path, "train/wav")
    aishell_test_dir = os.path.join(aishell_3_folder_path, "test/wav")
    wham_noise_train_metadata = os.path.join(wham_noise_folder_path, "metadata/noise_meta_tr.csv")
    wham_noise_test_metadata = os.path.join(wham_noise_folder_path, "metadata/noise_meta_tt.csv")

    wham_noise_train_dir = os.path.join(wham_noise_folder_path, "tr")
    wham_noise_test_dir = os.path.join(wham_noise_folder_path, "tt")

    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/test"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/train/clean"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/train/noisy"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/train/ground_truth"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/test/clean"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/test/noisy"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path,"aishell3-2mix/test/ground_truth"), exist_ok=True)

    clean_metadata = []
    noisy_metadata = []
    already_use_clean = set({})
    already_use_noise = set({})
    noise_df = pd.read_csv(wham_noise_test_metadata,na_filter=False)
    test_speaker_list = os.listdir(aishell_test_dir)
    print(f'There are {len(test_speaker_list)} in test set')

    count = 0
    for speaker in test_speaker_list:
        print(f"process speaker {speaker}")
        wav_folder = os.path.join(aishell_test_dir,speaker)
        wav_list = os.listdir(wav_folder)
        rd.shuffle(wav_list)
        wav_list = wav_list[:len(wav_list)//2+1]
        for wav in wav_list:
            wav2,speaker2  = choose_random_segment(aishell_test_dir,test_speaker_list,speaker)
            
             
            noise = get_random_noise(noise_df)

            _id_clean = tuple(sorted([wav,wav2]))
            _id_noisy = tuple(sorted([wav,wav2,noise]))

            while _id_clean in already_use_clean or _id_noisy in already_use_noise:
                wav2, speaker2 = choose_random_segment(aishell_test_dir,test_speaker_list,speaker)
                noise = get_random_noise(noise_df)

                _id_clean = tuple(sorted([wav,wav2]))
                _id_noisy = tuple(sorted([wav,wav2,noise]))

            already_use_clean.add(_id_clean)
            already_use_noise.add(_id_noisy)
            wav1_tensor, rate = torchaudio.load(os.path.join(wav_folder,wav))
            wav1_tensor = torchaudio.functional.resample(wav1_tensor,rate,16000)
            wav2_folder = os.path.join(aishell_test_dir,speaker2)
            wav2_tensor, rate = torchaudio.load(os.path.join(wav2_folder,wav2))
            wav2_tensor = torchaudio.functional.resample(wav2_tensor,rate,16000)

            length = max(wav1_tensor.shape[-1],wav2_tensor.shape[-1])
            if wav1_tensor.shape[-1] < length:
                r = int(math.ceil((length)/wav1_tensor.shape[-1]))
                wav1_tensor = torch.cat([wav1_tensor]*r,dim = -1)
            if wav2_tensor.shape[-1] < length:
                r = int(math.ceil((length)/wav2_tensor.shape[-1]))
                wav2_tensor = torch.cat([wav2_tensor]*r,dim = -1)
            wav1_tensor = wav1_tensor[:,:length]
            wav2_tensor = wav2_tensor[:,:length]

            noise_tensor, rate = torchaudio.load(os.path.join(wham_noise_test_dir,noise))
            noise_tensor = noise_tensor[0:1,:]
            noise_tensor = torchaudio.functional.resample(noise_tensor,rate,16000)

            if noise_tensor.shape[-1] < length:
                r = int(math.ceil((length)/noise_tensor.shape[-1]))
                noise_tensor = torch.cat([noise_tensor]*r,dim=-1)
            noise_tensor = noise_tensor[:,:length]
            mix_rate = rd.uniform(-5,10)
            noise_rate = rd.uniform(0,15)

            clean_metadata.append({
                "mix": f"_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav",
                "source1": f"_{count}_"+wav,
                "source2": f"_{count}_"+wav2,
                "mix_rate": mix_rate,
                "length": wav1_tensor.shape[-1]/16000
            })

            noisy_metadata.append({
                "mix": f"_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav",
                "source1": f"_{count}_"+wav,
                "source2": f"_{count}_"+wav2,
                "mix_rate": mix_rate,
                "noise": noise,
                "noise_rate": noise_rate,
                "length":wav1_tensor.shape[-1]/16000
            })

            mix = torchaudio.functional.add_noise(wav1_tensor,wav2_tensor,torch.tensor([mix_rate]))
            mix_noise = torchaudio.functional.add_noise(mix,noise_tensor,torch.tensor([noise_rate]))
            print(mix.shape,mix_noise.shape,wav1_tensor.shape,wav2_tensor.shape)
            torchaudio.save(os.path.join(output_folder_path,f'aishell3-2mix/test/ground_truth/_{count}_{wav}'),wav1_tensor,sample_rate=16000)
            torchaudio.save(os.path.join(output_folder_path,f'aishell3-2mix/test/ground_truth/_{count}_{wav2}'),wav2_tensor,sample_rate=16000)
            torchaudio.save(os.path.join(output_folder_path,f"aishell3-2mix/test/clean/_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav"),mix,sample_rate=16000)
            torchaudio.save(os.path.join(output_folder_path,f"aishell3-2mix/test/noisy/_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav"),mix_noise,sample_rate=16000)
            count +=1
    
    df_clean = pd.DataFrame(clean_metadata)
    df_noise = pd.DataFrame(noisy_metadata)
    df_clean.to_csv(os.path.join(output_folder_path,f"aishell3-2mix/test/clean_test_metadata.csv"))
    df_noise.to_csv(os.path.join(output_folder_path,f"aishell3-2mix/test/noisy_test_metadata.csv"))

    ### max mode
    ## train
    clean_metadata = []
    noisy_metadata = []
    already_use_clean = set({})
    already_use_noise = set({})
    noise_df = pd.read_csv(wham_noise_train_metadata,na_filter=False)
    train_speaker_list = os.listdir(aishell_train_dir)
    print(f'There are {len(train_speaker_list)} in train set')
    count = 0
    for speaker in train_speaker_list:
        wav_folder = os.path.join(aishell_train_dir,speaker)
        wav_list = os.listdir(wav_folder)
        for wav in wav_list:
            wav2,speaker2  = choose_random_segment(aishell_train_dir,train_speaker_list,speaker)
            noise = get_random_noise(noise_df)

            _id_clean = tuple(sorted([wav,wav2]))
            _id_noisy = tuple(sorted([wav,wav2,noise]))

            while _id_clean in already_use_clean or _id_noisy in already_use_noise:
                wav2,speaker2 = choose_random_segment(aishell_train_dir,train_speaker_list,speaker)
                noise = get_random_noise(noise_df)

                _id_clean = tuple(sorted([wav,wav2]))
                _id_noisy = tuple(sorted([wav,wav2,noise]))

            already_use_clean.add(_id_clean)
            already_use_noise.add(_id_noisy)
            wav1_tensor, rate = torchaudio.load(os.path.join(wav_folder,wav))
            wav1_tensor = torchaudio.functional.resample(wav1_tensor,rate,16000)
            wav2_folder = os.path.join(aishell_train_dir, speaker2)
            wav2_tensor, rate = torchaudio.load(os.path.join(wav2_folder,wav2))
            wav2_tensor = torchaudio.functional.resample(wav2_tensor,rate,16000)

            length = max(wav1_tensor.shape[-1],wav2_tensor.shape[-1])
            if wav1_tensor.shape[-1] < 16000*length:
                r = int(math.ceil((length)/wav1_tensor.shape[-1]))
                wav1_tensor = torch.cat([wav1_tensor]*r,dim = -1)
            if wav2_tensor.shape[-1] < length:
                r = int(math.ceil((length)/wav2_tensor.shape[-1]))
                wav2_tensor = torch.cat([wav2_tensor]*r,dim = -1)
            wav1_tensor = wav1_tensor[:,:length]
            wav2_tensor = wav2_tensor[:,:length]

            noise_tensor, rate = torchaudio.load(os.path.join(wham_noise_train_dir,noise))
            noise_tensor = noise_tensor[0:1,:]
            noise_tensor = torchaudio.functional.resample(noise_tensor,rate,16000)

            if noise_tensor.shape[-1] < length:
                r = int(math.ceil((length)/noise_tensor.shape[-1]))
                noise_tensor = torch.cat([noise_tensor]*r,dim=-1)
            noise_tensor = noise_tensor[:,:length]
            mix_rate = rd.uniform(-5,10)
            noise_rate = rd.uniform(0,15)

            clean_metadata.append({
                "mix": f"_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav",
                "source1": f"_{count}_"+wav,
                "source2": f"_{count}_"+wav2,
                "mix_rate": mix_rate,
                "length": wav1_tensor.shape[-1]/16000
            })

            noisy_metadata.append({
                "mix": f"_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav",
                "source1": f"_{count}_"+wav,
                "source2": f"_{count}_"+wav2,
                "mix_rate": mix_rate,
                "noise": noise,
                "noise_rate": noise_rate,
                "length":wav1_tensor.shape[-1]/16000
            })

            mix = torchaudio.functional.add_noise(wav1_tensor,wav2_tensor,torch.tensor([mix_rate]))
            mix_noise = torchaudio.functional.add_noise(mix,noise_tensor,torch.tensor([noise_rate]))
            torchaudio.save(os.path.join(output_folder_path,f"aishell3-2mix/train/ground_truth/_{count}_{wav}"),wav1_tensor,sample_rate=16000)
            torchaudio.save(os.path.join(output_folder_path,f"aishell3-2mix/train/ground_truth/_{count}_{wav2}"),wav2_tensor,sample_rate=16000)
            torchaudio.save(os.path.join(output_folder_path,f"aishell3-2mix/train/clean/_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav"),mix,sample_rate=16000)
            torchaudio.save(os.path.join(output_folder_path,f"aishell3-2mix/train/noisy/_{count}_{wav.replace('.wav','')}_{wav2.replace('.wav','')}.wav"),mix_noise,sample_rate=16000)
            count +=1
    df_clean = pd.DataFrame(clean_metadata)
    df_noise = pd.DataFrame(noisy_metadata)
    df_clean.to_csv(os.path.join(output_folder_path,f"aishell3-2mix/train/clean_train_metadata.csv"))
    df_noise.to_csv(os.path.join(output_folder_path,f"aishell3-2mix/noisy_train_metadata.csv"))
    print('finish')



class AiShell3mixDataset(Dataset):
    def __init__(self, path: str, mode: str, sample_rate=8000, segment_length=4):
        super().__init__()
        assert mode in ['noisy', 'clean'], "mode only accept noisy or clean"
        self.mix = os.path.join(path, mode)
        self.ground_truth = os.path.join(path, 'ground_truth')
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        metadata_file = list(filter(lambda x: x.startswith(
            mode) and x.endswith('csv'), os.listdir(path)))[0]
        self.metadata = pd.read_csv(os.path.join(path,metadata_file))
        current_length = len(self.metadata)
        self.metadata = self.metadata[self.metadata['length'] >= self.segment_length]
        print(f"Drop {len(self.metadata) - current_length} segment because length less than {segment_length} seconds")
    def __len__(self):
        return 2*len(self.metadata)

    def __getitem__(self, index):
        src_idx = index // 2

        row = self.metadata.iloc[src_idx]
        mix = row['mix']
        src = row['source1'] if index % 2 == 1 else row['source2']
        src_16k,r = torchaudio.load(os.path.join(self.ground_truth,src))
        src_16k = src_16k[:,:self.segment_length*r]
        src = torchaudio.functional.resample(src_16k,r,self.sample_rate)

        mix, r = torchaudio.load(os.path.join(self.mix,mix))
        mix = mix[:,:self.segment_length*r]
        
        mix = torchaudio.functional.resample(mix,r,self.sample_rate)
        return mix.squeeze(), src.squeeze(), src_16k.squeeze()