#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import os.path as osp
import sys
import glob
import json
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--root_dir', type=str, default="data")
  parser.add_argument('--pids', type=str, default="configs/train_pids.txt")
  parser.add_argument('--pid_list', type=str, nargs="*", default=[])
  parser.add_argument('--nb_cols', type=int, default=20)
  parser.add_argument('--size', type=int, default=112+56)
  parser.add_argument('--out_path', type=str, default="out.jpg")
  parser.add_argument('--stats', action='store_true')
  parser.add_argument('--patch', action='store_true')
  parser.add_argument('--video', action='store_true')
  parser.add_argument('--summary', action='store_true')
  parser.add_argument('--frame_track_id', type=str)
  args = parser.parse_args()

  image_dir = osp.join(args.root_dir, "epic_kitchens", "rgb_frames")
  root_anno_dir = osp.join(args.root_dir, "vott")

  if len(args.pid_list) == 0:
    with open(args.pids) as f:
      pid_list = [x.strip("\n") for x in f.readlines()]
  else:
    pid_list = args.pid_list

  if args.stats:
    # Statistics
    nb_instances, nb_tracks, nb_frames = 0, 0, 0
    nb_frames_list, nb_tracks_list = [], []
    for pid in pid_list:
      anno_dir = osp.join(root_anno_dir, pid)
      for anno_path in list(sorted(glob.glob(osp.join(anno_dir, "*.json")))):

        with open(anno_path) as f:
          vott_dict = json.load(f)

        found = False
        ntr = 0
        dets, prev_frame_num, prev_vid = [], None, None
        for key, info in vott_dict["frames"].items():
          if len(info) == 0:
            continue
          vid = "_".join(key.split("_")[:2])
          frame_num = int(key.split("_")[2][:-4])

          if prev_vid is not None and (prev_vid != vid or frame_num - prev_frame_num > 60):
            nb_tracks += 1
            ntr += 1
            nb_frames += len(dets)
            nb_frames_list.append(len(dets))
            found = True
            dets = []
          det = [vid, frame_num, info[0]["x1"], info[0]["y1"], info[0]["x2"], info[0]["y2"]]
          dets.append(det)
          prev_vid = vid
          prev_frame_num = frame_num

        if len(dets) > 0:
          nb_tracks += 1
          ntr += 1
          nb_frames += len(dets)
          nb_frames_list.append(len(dets))
          found = True

        if found:
          nb_instances += 1
          nb_tracks_list.append(ntr)

    nb_frames_list = np.array(nb_frames_list)
    nb_frames_list[nb_frames_list > 100] = 100
    print(len(pid_list), nb_instances, nb_tracks, nb_frames)

    plt.figure()
    plt.hist(nb_frames_list, range=(1, 101), bins=100)
    plt.yscale('log')
    plt.xlabel("Number of Frames")
    plt.ylabel("Frequency")
    plt.savefig("nb_frames_per_track.png")
    plt.close()

    plt.figure()
    plt.hist(nb_tracks_list, range=(1, 101), bins=100)
    plt.xlabel("Number of Tracks")
    plt.ylabel("Frequency")
    plt.savefig("nb_tracks_per_instance.png")
    plt.close()

  if args.patch:
    for pid in pid_list:
      anno_dir = osp.join(root_anno_dir, pid)
      for anno_path in list(sorted(glob.glob(osp.join(anno_dir, "*.json")))):

        track_id = osp.splitext(osp.basename(anno_path))[0]
        pid = track_id[:3]
        if args.frame_track_id is not None and args.frame_track_id != track_id:
          continue

        with open(anno_path) as f:
          vott_dict = json.load(f)

        dets_list = []

        if args.frame_track_id is not None:
          out_dir = osp.join("data", "patch_outputs_selected", pid, track_id)
        else:
          out_dir = osp.join("data", "patch_outputs", pid, track_id)
        os.makedirs(out_dir, exist_ok=True)

        frame_cnt = 0

        dets, prev_frame_num, prev_vid = [], None, None
        for key, info in vott_dict["frames"].items():
          if len(info) == 0:
            continue
          vid = "_".join(key.split("_")[:2])
          frame_num = int(key.split("_")[2][:-4])
          if prev_vid is not None and (prev_vid != vid or frame_num - prev_frame_num > 60):
            dets_list.append(dets)
            dets = []

          det = [vid, frame_num, info[0]["x1"], info[0]["y1"], info[0]["x2"], info[0]["y2"]]
          dets.append(det)
          prev_vid = vid
          prev_frame_num = frame_num

        if len(dets) > 0:
          dets_list.append(dets)

        nb_frames_per_track = 5 if len(dets_list) <= 20 else 3

        for tidx, dets in enumerate(dets_list):
          nb_frames = len(dets)
          if nb_frames > nb_frames_per_track:
            selected_idxs = np.linspace(0, nb_frames-1, nb_frames_per_track, dtype=int).tolist()
          else:
            selected_idxs = list(range(nb_frames))

          for idx in selected_idxs:
            vid, frame_num, x1, y1, x2, y2 = dets[idx]

            impath = osp.join(image_dir, pid, vid, f"frame_{frame_num:010d}.jpg")
            img = cv2.imread(impath)

            pad = 0.1
            w, h = x2 - x1, y2 - y1
            px1 = max(0, int(x1 - w * pad))
            py1 = max(0, int(y1 - h * pad))
            px2 = min(img.shape[1], int(x2 + w * pad))
            py2 = min(img.shape[0], int(y2 + h * pad))

            if px2 - px1 > py2 - py1:
              w = px2 - px1
              ny1 = max(0, int((py1 + py2) / 2 - w / 2))
              ny2 = min(int(py1 + w), img.shape[0])
              nx1, nx2 = px1, px2
            else:
              h = py2 - py1
              nx1 = max(0, int((px1 + px2) / 2 - h / 2))
              nx2 = min(int(px1 + h), img.shape[1])
              ny1, ny2 = py1, py2

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            patch = cv2.resize(img[ny1:ny2, nx1:nx2], (args.size, args.size))
            cv2.imwrite(osp.join(out_dir, f"{track_id}_{tidx}_{idx}.jpg"), patch)
            if args.frame_track_id is not None:
              cv2.imwrite(osp.join(out_dir, f"{track_id}_{tidx}_{idx}_frame.jpg"), img)
            frame_cnt += 1

        print(track_id, frame_cnt)

  if args.video:
    for pid in pid_list:
      anno_dir = osp.join(root_anno_dir, pid)

      pad = 0.25
      fps = 10
      out_size = (args.size, args.size)
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      out_dir = osp.join("data", "patch_videos", pid)
      print(f"Output dir: {out_dir}")
      os.makedirs(out_dir, exist_ok=True)

      for anno_path in list(sorted(glob.glob(osp.join(anno_dir, "*.json")))):

        track_id = osp.splitext(osp.basename(anno_path))[0]
        pid = track_id[:3]
        if args.frame_track_id is not None and args.frame_track_id != track_id:
          continue

        with open(anno_path) as f:
          vott_dict = json.load(f)

        dets_list = []
        frame_cnt = 0

        dets, prev_frame_num, prev_vid = [], None, None
        for key, info in vott_dict["frames"].items():
          if len(info) == 0:
            continue
          vid = "_".join(key.split("_")[:2])
          frame_num = int(key.split("_")[2][:-4])
          if prev_vid is not None and (prev_vid != vid or frame_num - prev_frame_num > 60):
            dets_list.append(dets)
            dets = []

          det = [vid, frame_num, info[0]["x1"], info[0]["y1"], info[0]["x2"], info[0]["y2"]]
          dets.append(det)
          prev_vid = vid
          prev_frame_num = frame_num

        if len(dets) > 0:
          dets_list.append(dets)

        #nb_frames_per_track = 5 if len(dets_list) <= 20 else 3
        #nb_frames_per_track = len(dets_list)
        if np.sum([len(x) for x in dets_list]) < 10:
          continue

        writer = cv2.VideoWriter(osp.join(out_dir, f"{track_id}.mp4"), fourcc, fps, out_size)

        for tidx, dets in enumerate(dets_list):
          nb_frames = len(dets)
          selected_idxs = list(range(nb_frames))

          for idx in selected_idxs:
            vid, frame_num, x1, y1, x2, y2 = dets[idx]

            impath = osp.join(image_dir, pid, vid, f"frame_{frame_num:010d}.jpg")
            img = cv2.imread(impath)

            w, h = x2 - x1, y2 - y1
            px1 = max(0, int(x1 - w * pad))
            py1 = max(0, int(y1 - h * pad))
            px2 = min(img.shape[1], int(x2 + w * pad))
            py2 = min(img.shape[0], int(y2 + h * pad))

            if px2 - px1 > py2 - py1:
              w = px2 - px1
              ny1 = max(0, int((py1 + py2) / 2 - w / 2))
              ny2 = min(int(py1 + w), img.shape[0])
              nx1, nx2 = px1, px2
            else:
              h = py2 - py1
              nx1 = max(0, int((px1 + px2) / 2 - h / 2))
              nx2 = min(int(px1 + h), img.shape[1])
              ny1, ny2 = py1, py2

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            patch = cv2.resize(img[ny1:ny2, nx1:nx2], (args.size, args.size))
            writer.write(patch)
            frame_cnt += 1

        print(track_id, frame_cnt)
        writer.release()

  if args.summary:
    # Pick up one image per cluster
    img_list = []
    for pid in pid_list:
      anno_dir = osp.join(root_anno_dir, pid)
      print(anno_dir)
      for anno_path in list(sorted(glob.glob(osp.join(anno_dir, "*.json")))):

        track_id = osp.splitext(osp.basename(anno_path))[0]

        with open(anno_path) as f:
          vott_dict = json.load(f)

        nb_frames = len(vott_dict["frames"])

        for key, info in vott_dict["frames"].items():
          vid = "_".join(key.split("_")[:2])
          frame_num = int(key.split("_")[2][:-4])
          x1, y1, x2, y2 = int(info[0]["x1"]), int(info[0]["y1"]), int(info[0]["x2"]), int(info[0]["y2"])

          impath = osp.join(image_dir, pid, vid, f"frame_{frame_num:010d}.jpg")
          img = cv2.imread(impath)

          pad = 0.1
          w, h = x2 - x1, y2 - y1
          px1 = max(0, int(x1 - w * pad))
          py1 = max(0, int(y1 - h * pad))
          px2 = min(img.shape[1], int(x2 + w * pad))
          py2 = min(img.shape[0], int(y2 + h * pad))

          if px2 - px1 > py2 - py1:
            w = px2 - px1
            ny1 = max(0, int((py1 + py2) / 2 - w / 2))
            ny2 = min(int(py1 + w), img.shape[0])
            nx1, nx2 = px1, px2
          else:
            h = py2 - py1
            nx1 = max(0, int((px1 + px2) / 2 - h / 2))
            nx2 = min(int(px1 + h), img.shape[1])
            ny1, ny2 = py1, py2

          cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
          img = cv2.resize(img[ny1:ny2, nx1:nx2], (args.size, args.size))

          #cv2.putText(img, track_id, (0, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_4)
          #cv2.putText(img, str(nb_frames), (0, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_4)

          img_list.append(img)
          break

    nb_cols = args.nb_cols
    nb_rows = (len(img_list) - 1) // nb_cols + 1

    out_img = np.ones((args.size * nb_rows, args.size * nb_cols, 3), dtype=np.uint8) * 255
    print(len(img_list), nb_rows, nb_cols, args.out_path)
    for idx, img in enumerate(img_list):

      h, w = idx // nb_cols, idx % nb_cols
      out_img[h*args.size:(h+1)*args.size, w*args.size:(w+1)*args.size] = img

    cv2.imwrite(args.out_path, out_img)


if __name__ == "__main__":
  main()
