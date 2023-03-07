import os

BIN_DIR = "/data/yesheng/3D-Scene/NeuRIS"
DIR_MVG_BUILD = "/data/yesheng/3D-Scene/code/openMVG_Build"
DIR_MVS_BUILD = "/data/yesheng/3D-Scene/code/openMVS_build"

# normal path
dir_snu_code = '/data/yesheng/3D-Scene/snucode' # directory of code
path_snu_pth = '/data/yesheng/3D-Scene/snucode/checkpoints/scannet.pt'
assert os.path.exists(path_snu_pth)

dir_tiltedsn_code = '/path/tiltedsn_code'
dir_tiltedsn_ckpt = '/path/tiltedsn_ckpt' # directory of pretrained model
path_tiltedsn_pth_pfpn = f"{dir_tiltedsn_ckpt}/PFPN_SR_full/model-best.ckpt"
path_tiltedsn_pth_sr = f"{dir_tiltedsn_ckpt}/SR_only/model-latest.ckpt"
# assert os.path.exists(path_tiltedsn_pth_sr)

# used scenes
names_scenes_neuris = ['N11']
names_scenes_manhattansdf = []
lis_name_scenes = names_scenes_neuris + names_scenes_manhattansdf

# update training/test split
names_scenes_neuris_remove = ['scene0009', 'scene0085', 'scene0114',
                        'scene0603', 'scene0617', 'scene0625',
                        'scene0721', 'scene0771']
names_scenes_manhattansdf_remove = ['scene0050', 'scene0084', 
                                'scene0580', 'scene0616']
lis_name_scenes_remove = names_scenes_neuris_remove + names_scenes_manhattansdf_remove
