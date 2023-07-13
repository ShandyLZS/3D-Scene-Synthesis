#  Visualize predictions
#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
from pathlib import Path
import argparse
import numpy as np
import trimesh
from utils.threed_front import Threed_Front_Config
import h5py
from utils.threed_front.vis.vis_classes import VIS_3DFRONT_RESULT
from utils.threed_front.tools.utils import project_insts_to_2d
from utils.threed_front.vis.vis_classes import VIS_3DFRONT_2D
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a 3D-FRONT gt sample.")
    parser.add_argument("--dump_dir", type=str, default='demo/3D-Front/output',
                        help="The directory of dumped results.")
    parser.add_argument("--sample_name", type=str, default='bed_8f90151a-098e-4156-b438-771584a19182_MasterBedroom-13855_89',
                        help="give the sample_name to visualize.")
    return parser.parse_args()


def read_pred_data(pred_file, save_mesh_dir=None):
    pred_data = np.load(pred_file)
    box3ds = np.concatenate([pred_data['centers'], pred_data['sizes']], axis=-1)
    box3ds = np.pad(box3ds, ((0, 0), (0, 1)))
    category_ids = pred_data['category_ids']
    sample_names = pred_data['sample_names']
    mesh_vertices = pred_data['mesh_vertices']
    mesh_faces = pred_data['mesh_faces']
    inst_mesh_files = []
    if save_mesh_dir is not None:
        color_palette = np.array(sns.color_palette('hls', len(dataset_config.label_names)))
        for inst_id, (vertices, faces) in enumerate(zip(mesh_vertices, mesh_faces)):
            color = color_palette[category_ids[inst_id]]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, vertex_colors=color)
            save_file = save_mesh_dir.joinpath('%d.obj' % inst_id)
            mesh.export(save_file)
            inst_mesh_files.append(str(save_file))

    return {'box3ds': box3ds, 'category_ids': category_ids, 'sample_names': sample_names.tolist(),
            'mesh_files': inst_mesh_files}


def read_gt_data(gt_sample_names):
    cam_Ts = []
    room_imgs = []
    instance_attrs = []
    class_maps = []
    projected_inst_boxes = []
    layout_boxes = []
    '''read data'''
    for gt_sample_name in gt_sample_names:
        for rm_type_dir in dataset_config.dump_dir_to_samples.iterdir():
            gt_file = rm_type_dir.joinpath(gt_sample_name + '.hdf5')
            if gt_file.is_file():
                break
        with h5py.File(gt_file, "r") as sample_data:
            room_uid = sample_data['room_uid'][0].decode("utf-8")
            room_type = sample_data['room_type'][0].decode("utf-8")
            colors = sample_data['colors'][:]
            layout_box = sample_data['layout_box'][:]
            cam_T = sample_data['cam_T'][:]
            class_segmap = sample_data['class_segmap'][:]
            inst_h5py = sample_data['inst_info']
            inst_info = []
            for inst_id in inst_h5py:
                inst = {}
                inst['bbox2d'] = inst_h5py[inst_id]['bbox2d'][:]
                inst['bbox3d'] = inst_h5py[inst_id]['bbox3d'][:]
                inst['category_id'] = inst_h5py[inst_id]['category_id'][0]
                inst['inst_mark'] = inst_h5py[inst_id]['inst_mark'][0].decode('utf-8')
                inst['uid'] = inst_h5py[inst_id]['uid'][0].decode('utf-8')
                inst['jid'] = inst_h5py[inst_id]['jid'][0].decode('utf-8')
                inst['mask'] = inst_h5py[inst_id]['mask'][:]
                inst['model_path'] = inst_h5py[inst_id]['model_path'][0].decode('utf-8')
                inst['room_id'] = inst_h5py[inst_id]['room_id'][0].decode('utf-8')
                inst_info.append(inst)

        '''Project objects to original cam poses'''
        projected_box2d_list = project_insts_to_2d(inst_info, dataset_config.cam_K, cam_T)

        cam_Ts.append(cam_T)
        room_imgs.append(colors)
        instance_attrs.append(inst_info)
        class_maps.append(class_segmap)
        projected_inst_boxes.append(projected_box2d_list)
        layout_boxes.append(layout_box)

    return {'cam_Ts': cam_Ts, 'room_imgs': room_imgs,
            'projected_inst_boxes': projected_inst_boxes, 'instance_attrs': instance_attrs,
            'class_maps': class_maps, 'layout_boxes': layout_boxes}


if __name__ == '__main__':
    args = parse_args()
    dataset_config = Threed_Front_Config()

    '''load gt and pred data'''
    pred_file = Path(args.dump_dir).joinpath(args.sample_name + '.npz')
    room_type = args.sample_name.split('_')[0]
    dataset_config.init_generic_categories_by_room_type(room_type)

    '''read pred data'''
    save_mesh_dir = Path('./temp').joinpath(args.sample_name)
    if not save_mesh_dir.exists():
        save_mesh_dir.mkdir(parents=True)
    pred_data = read_pred_data(pred_file, save_mesh_dir=save_mesh_dir)
    sample_names = set(pred_data['sample_names'])

    '''read gt data'''
    gt_data = read_gt_data(sample_names)

    if len(sample_names) == 1:
        print('Finetuning only using a single image.')
        n_objects_in_scene = len(gt_data['instance_attrs'][0])
        pred_data['box3ds'] = pred_data['box3ds'][:n_objects_in_scene]
        pred_data['category_ids'] = pred_data['category_ids'][:n_objects_in_scene]

    cam_K = dataset_config.cam_K

    '''visualize results'''
    # vis gt
    viser_2D = VIS_3DFRONT_2D(color_maps=gt_data['room_imgs'], inst_info=gt_data['instance_attrs'],
                              cls_maps=gt_data['class_maps'], class_names=dataset_config.label_names,
                              projected_inst_boxes=gt_data['projected_inst_boxes'])
    viser_2D.draw_colors()
    viser_2D.draw_cls_maps()
    viser_2D.draw_box2d_from_3d()
    viser_2D.draw_inst_maps(type=('mask'))

    # vis prediction
    pred_projected_inst_boxes = []
    for cam_T in gt_data['cam_Ts']:
        pred_box3ds = [{'bbox3d': box3d} for box3d in pred_data['box3ds']]
        pred_projected_inst_boxes.append(project_insts_to_2d(pred_box3ds, dataset_config.cam_K, cam_T))
    viser = VIS_3DFRONT_RESULT(cam_K=cam_K, cam_Ts=np.array(gt_data['cam_Ts']),
                               box2ds=None,
                               masks=None,
                               box3ds=pred_data['box3ds'],
                               mesh_files=pred_data['mesh_files'],
                               category_ids=pred_data['category_ids'], class_names=dataset_config.label_names,
                               projected_inst_boxes=pred_projected_inst_boxes, color_maps=gt_data['room_imgs'])
    viser.draw_box2d_from_3d()
    viser.visualize(view_id=0, type=[])
