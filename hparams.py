RESNET = {
    "name": "resnet",
    "num_classes": 10,

    # audio
    "sr": 22050,
    "duration": 4.0,
    "crop_duration": 3.0,
    "n_mels": 128,
    "n_fft": 1024,
    "hop": 256,

    # train
    "batch_size": 32,
    "epochs": 50,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,

    # mixup
    "mixup_alpha": 0.27,
    "use_mixup": True,

    # specaug
    "spec_aug": True,
    "time_mask": 18,
    "freq_mask": 8,

    "patience": 7,
    "num_workers": 0,
    "ema_decay": 0.995,

    # cache dir (keep per-model)
    "cache_dir": "mel_cache_resnet18_no_time_aug",
}

# CRNN hyperparams (match your script)
CRNN = {
    "name": "crnn",
    "num_classes": 10,

    # audio
    "sample_rate": 22050,
    "duration": 4.0,
    "crop_duration": 3.0,
    "n_mels": 128,
    "hop_length": 256,
    "n_fft_list": [400, 1024, 2048],

    # train
    "batch_size": 32,
    "epochs": 40,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.10,

    # mixup
    "mixup_alpha": 0.16,
    "use_mixup": True,

    # aug probs
    "time_mask_max": 32,
    "freq_mask_max": 16,
    "time_mask_num": 2,
    "freq_mask_num": 2,
    "noise_std_low": 0.0,
    "noise_std_high": 0.02,
    "noise_prob": 0.4,
    "spec_aug_prob": 0.35,

    "patience": 5,
    "num_workers": 2,
    "ema_decay": 0.995,

    "cache_dir": "mel_cache_multires_crnn_1",
}
