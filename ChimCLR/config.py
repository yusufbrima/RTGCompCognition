from pathlib import Path

audio = dict(
    FRAME_LENGHT =  1024,
    SAMPLE_RATE =  44100,
    HOP_LENGTH =  512,
    CHANNELS = 1
)


hyperparams = dict(
    LEARNING_RATE =  0.001,
    BATCH_SIZE =  8,
    EPOCHS =  20
)

model = dict(
    num_classes =  10
)



data = dict(
    INPUT_PATH = Path('/net/store/cbc/projects/Pan troglodytes/audio_PH_dataset/PH/good_data/'),
    file_path =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango.npz',
    file_path2 =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/clip_loango_resized.npz',
    BASE_PATH =  '/net/store/cv/users/ybrima/scratch/data/Luango_Speaker/',
    SHUFFLE_BUFFER_SIZE = 50,
    BATCH_SIZE = 8,

)


figures = dict(
    figpath =  './Figures'
)