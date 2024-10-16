# Agent-to-Sim: Learning Interactive Behavior from Casual Videos
#### [**Project**](https://gengshan-y.github.io/agent2sim-www/) | [**Paper**](https://gengshan-y.github.io/agent2sim-www/materials/ATS-sep9_compressed.pdf) 

## TODOs
- [ ] Splat refinement code merge (see [Lab4d-GS](https://github.com/lab4d-org/lab4d/tree/lab4dgs) for reference)
- [ ] Clean up

## Installation
```
git clone git@github.com:facebookresearch/agent2sim.git --recursive
git submodule update --init --recursive
cd lab4d
```
Follow [this](https://lab4d-org.github.io/lab4d/get_started/) to install lab4d in a new conda environment.

Then run
```
pip install networkx==2.5
```
to resolve the remaining compatibility issue.



## 4D Reconstruction
Make sure you are in the `agent2sim/lab4d` dir.
### Rendering
Download checkpoints
Cat:
```
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/rgeytrcy0498b90u8sufu/log-cat-pikachu-2024-08-v2-compose-ft.zip?rlkey=7uoco9701r15i9e222ujlh1kt&st=xyrsoj4j&dl=0"
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/m07qu1megz5f98xlv0f38/log-cat-pikachu-2024-08-v2-bg-adapt4.zip?rlkey=a1vgou82aw8kqnup2tqfd2qvq&st=wicrkz0p&dl=0"
```

Human:
```
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/3my433p62szmbat3bj94m/log-human-2024-05-compose-ft.zip?rlkey=3hv6f3wk06t4asrw8stqwfjdp&st=yiuh6hqm&dl=0"

bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/vbvj5dy7fkbme0ulbwioc/log-human-2024-05-bg-adapt3.zip?rlkey=qvtq0ba7wbidfpqfvj4wrzz6i&st=f591hrcn&dl=0"
```

Extrack meshes and render videos
```
python lab4d/export.py --flagfile=logdir/cat-pikachu-2024-08-v2-compose-ft/opts.log --load_suffix latest --inst_id 1 --vis_thresh -20 --grid_size 128 --data_prefix full 0
```

Volumen rendering
```
python lab4d/render.py --flagfile=logdir/cat-pikachu-2024-08-v2-compose-ft/opts.log --load_suffix latest --viewpoint ref --inst_id 1 --render_res 128 --n_depth 256 --freeze_id 0 --num_frames 2
```

### Optimization
Download training data
```
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/x97tjjxfblmehpb1spow6/polycam_all.zip?rlkey=r84ef8rhwn66pzllqdceezst7&st=blpalij7&dl=0"
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/zztig9emr7nt5rh82hibv/log-predictor-bunny-scene.zip?rlkey=n1jjk77hsn0gboa4jl32zx06s&st=78ljtgui&dl=0"
```

Formatting data
```
python projects/csim/record3d_to_lab4d.py
```

Run optimization
```
bash projects/csim/run_multi.sh Feb14at5-55PM-poly bunny 0,1
```
Results will be saved to `logdir/bunny-compose-ft/`.


## Motion Generation
```
cd gdmdm
ln -s ../lab4d/logdir logdir
ln -s ../lab4d/database database
ln -s ../lab4d/tmp tmp
```

### Testing
Download checkpoints
```
bash ../lab4d/scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/k6mlktvu8qy0ndebho56b/log-cat-pikachu-2024-07-compose-ft-b128-past-old.zip?rlkey=qydtsgqzh9xamafy0tcdi3wzh&st=axbwi9ut&dl=0"
bash ../lab4d/scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/9l43di09yxll69o52ban7/log-cat-pikachu-2024-07-compose-ft.zip?rlkey=9whmlmisycx5blmf56y9jazo1&st=fbbztge8&dl=0"
```

Run interactive gui
```
python long_video_two_agents.py --load_logname cat-pikachu-2024-07-compose-ft --logname_gd b128-past-old --sample_idx 0 --eval_batch_size 1  --load_suffix latest
```

### Training

### Prepare training data
#### Option 1: download processed data
```
wget "https://www.dropbox.com/scl/fi/9jkme44gesqifaxq85ta1/cat-pikachu-2024-08-v2-compose-ft-train-L64-S1.pkl?rlkey=iy6t0s7afle9siaowf7bp3up4&st=03cnk71l&dl=0" -O data/motion/cat-pikachu-2024-08-v2-compose-ft-train-L64-S1.pkl

```

#### Option 2: extract motion from 4D reconstruction
```
python generate_data.py --in_path "../lab4d/logdir/bunny-compose-ft/export_*"
```
The motion data will be saved to `database/motion`.

### Visualization

https://github.com/user-attachments/assets/63249dc2-9add-407f-8fe7-c1d1b24c2631

You can visualize motion data by
```
python visualize_dataset.py --load_logname cat-pikachu-2024-07-compose-ft
```
or 
```
python visualize_dataset.py --load_logname bunny-compose-ft
```

### Training
```
bash train.sh cat-pikachu-2024-08-v2-compose-ft b128 128 1
```


## Evaluatiom
Make sure you are in the `agent2sim/lab4d` dir.
Download data
```
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/o76jq3bdd9xxvcxwal5q2/aux.zip?rlkey=54b4ku3ae538pkn9wve2j95wo&st=4q9tawcn&dl=0"
bash scripts/download_unzip.sh "https://www.dropbox.com/scl/fi/gkmyqmv6xvgyxtlz91jfn/config.zip?rlkey=neg7zn4qsd4qv8wg918d1rdap&st=b3ahcprt&dl=0"
```

Registration
```
python projects/csim/scripts/kps_to_extrinsics.py
```

4D reconstrcution
```
python projects/csim/scripts/eval_4drecon.py --flagfile=logdir-neurips-aba/cat-pikachu-2024-08-v2-compose-ft2/opts.log
```
Results should be consistent with Tab.4/5 of the paper.

## License
The majority of agent2sim is licensed CC-by-NC, however portions of the project are available under separate license terms: Lab4d is licensed MIT.
