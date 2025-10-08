python examples/data_preprocess/video_reason/video_holmes_multiturn_w_tool.py \
    --json_path /mnt/aws-lfs-01/shared/jingwang/PROJECTS/video_reason/eval/Video-Holmes/Benchmark/test_Video-Holmes.json\
    --video_dir /mnt/aws-lfs-01/shared/jingwang/PROJECTS/video_reason/eval/Video-Holmes/Benchmark/videos \
    --output_path /mnt/aws-lfs-01/shared/datasets/video_data/Video-Holmes/test.parquet


python examples/data_preprocess/video_reason/video_holmes_multiturn_w_tool.py \
    --json_path /mnt/aws-lfs-01/shared/jingwang/PROJECTS/video_reason/eval/Video-Holmes/Benchmark/train_Video-Holmes.json \
    --video_dir /mnt/aws-lfs-01/shared/jingwang/PROJECTS/video_reason/eval/Video-Holmes/Benchmark/videos \
    --output_path /mnt/aws-lfs-01/shared/datasets/video_data/Video-Holmes/train.parquet