#!/usr/bin/python

import os.path as osp
import sys
import argparse
import os, tempfile, glob, shutil
import global_variables

def render_object_view(model_file, azimuth, elevation, tilt, distance, output_dir, file_name):

    blank_file = osp.join(global_variables.g_blank_blend_file_path)
    render_code = osp.join(global_variables.g_render4cnn_root_folder, 'render_pipeline/render_model_views.py')

    # MK TEMP DIR
    temp_dirname = tempfile.mkdtemp()
    view_file = osp.join(temp_dirname, 'view.txt')
    view_fout = open(view_file,'w')
    view_fout.write(' '.join([str(azimuth), str(elevation), str(tilt), str(distance)]))
    view_fout.close()

    try:
        render_cmd = '%s %s -noaudio --background --python %s -- %s %s %s %s %s' % (global_variables.g_blender_executable_path, blank_file, render_code, model_file, file_name, 'xxx', view_file, temp_dirname)
        os.system(render_cmd)
        imgs = glob.glob(temp_dirname+'/*.png')
        shutil.move(imgs[0], output_dir)
    except:
        print('render failed. render_cmd: %s' % (render_cmd))

    # CLEAN UP
    shutil.rmtree(temp_dirname)


def render_object_views(model_file, view_files, output_dir, file_name):

    blank_file = osp.join(global_variables.g_blank_blend_file_path)
    render_code = osp.join(global_variables.g_render4cnn_root_folder, 'render_pipeline/render_model_views.py')

    # MK TEMP DIR
    temp_dirname = tempfile.mkdtemp()
    view_file = osp.join(temp_dirname, 'view.txt')
    view_fout = open(view_file,'w')
    view_fout.write(' '.join([str(azimuth), str(elevation), str(tilt), str(distance)]))
    view_fout.close()

    try:
        render_cmd = '%s %s -noaudio --background --python %s -- %s %s %s %s %s' % (global_variables.g_blender_executable_path, blank_file, render_code, model_file, file_name, 'xxx', view_file, temp_dirname)
        os.system(render_cmd)
        imgs = glob.glob(temp_dirname+'/*.png')
        shutil.move(imgs[0], output_dir)
    except:
        print('render failed. render_cmd: %s' % (render_cmd))

    # CLEAN UP
    shutil.rmtree(temp_dirname)



def make_model_file_name(class_folder, hash_dir):
    obj_path = os.path.join(class_folder, hash_dir, 'models')
    return os.path.join(obj_path, glob.glob1(obj_path, '*.obj')[0])


output_folder = os.path.join(global_variables.g_datasets_folder, 'RENDERS')
class_folder = os.path.join(global_variables.g_shapenet_root_folder, '02958343')
for hash_dir in os.listdir(class_folder):
    full_has_dir = os.path.join(class_folder, hash_dir)
    if os.path.isdir(full_has_dir):
        example_object_file = make_model_file_name(class_folder, hash_dir)
        assert os.path.isfile(example_object_file), "file: {} doesn't exist".format(example_object_file)
        render_object_view(example_object_file, 240.0, 25.0, 0, 2.4, output_folder, hash_dir)