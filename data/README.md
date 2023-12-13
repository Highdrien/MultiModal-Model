# PACO-CHEESE

Fichiers
```
├── list_files.csv
├── audio
│   ├── 1_channels # compressed
│   └── 2_channels # extracted from video because some missing files
├── transcr
│   ├── {dyad}_merge.eaf # contains {speaker}-Transcription, Syllabic Alignment, PoS Tagging from MarsaTag, turn, turn-actions
│   └── {dyad}_merge.csv # data & labels at ipu level
│   └── {dyad}_merge_words.csv # data & labels at word level
└── video
    ├── paco
    ├── cheese
    └── openface # videos were cropped to 950x1080 (0:950 and 970:1920) so as to keep only 1 face

```

Missing files / files with issues
* 'AWCG_CG': OpenFace Data missing for timestamp in [997.920:1033.680]
* 'AAOR_AA': missing 1_channel audio (available in 2_channels file)
* 'MDAD': no video (didn't consent to video being shared)
* 'JRGB': removed files because issue in transcription

Notes:
* 1_channel audios are in 'mp4' format, not an issue