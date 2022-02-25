from pathlib import Path

audio = dict(
    FRAME_LENGHT =  1024,
    SAMPLE_RATE =  44100,
    HOP_LENGTH =  512,
    CHANNELS = 1,
    DURATION = 1
)


hyperparams = dict(
    LEARNING_RATE =  0.003,
    BATCH_SIZE =  128,
    EPOCHS =  20
)

experiment = dict(
    NAME =  'Chimp'
)


data = dict(
    INPUT_PATH = Path('/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/Full_Data/good'),
    CONVERT_PATH = Path('/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/Full_Data/Converted'), 
    OUT_PATH = Path('/net/projects/scratch/winter/valid_until_31_July_2022/ybrima/data/Full_Data/Cleaned'),
    file_path =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango.npz',
    file_path_pan =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango_pan.npz',
    file_path2 =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango_resized.npz',
    BASE_PATH =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/',
    SHUFFLE_BUFFER_SIZE = 50,
    BATCH_SIZE = 8,

)

# data = dict(
#     INPUT_PATH = Path('/net/store/cv/users/ybrima/scratch/data/archive/16000_pcm_speeches/'), 
#     file_path =  '/net/store/cv/users/ybrima/scratch/data/archive/speeches.npz',
#     file_path2 =  '/net/store/cv/users/ybrima/scratch/data/archive/speeches_resized.npz',
#     BASE_PATH =  '/net/store/cv/users/ybrima/scratch/data/archive/16000_pcm_speeches/',
#     SHUFFLE_BUFFER_SIZE = 50,
#     BATCH_SIZE = 8,

# )
#

figures = dict(
    figpath =  './Figures'
)