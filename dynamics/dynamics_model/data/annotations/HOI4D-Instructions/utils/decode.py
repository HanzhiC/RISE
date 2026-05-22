import os
import subprocess
import sys

def decode_video(root):

    with open(
        "/home/wiss/chenh/mobile_manip/egoasis3D/data/annotations/HOI4D-Instructions/release.txt",
        "r",
    ) as f:
        rgb_list = [os.path.join(root, i.strip(),'align_rgb') for i in f.readlines()]
    # rgb_list = ["ZY20210800004/H4/C17/N23/S134/s04/T1"]
    # rgb_list = [os.path.join(root, i.strip(),'align_rgb') for i in rgb_list]
    for rgb in rgb_list:
        depth = rgb.replace('align_rgb','align_depth')
        rgb_video = os.path.join(rgb, "image.mp4")
        depth_video = os.path.join(depth, "depth_video.avi")

        cmd = """ ffmpeg -i {} -f image2 -start_number 0 -vf fps=fps=15,scale=iw/4:ih/4:flags=lanczos -qscale:v 2 {}/%05d.{} -loglevel quiet """.format(
            rgb_video, rgb, "jpg"
        )

        # print(cmd)
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            log.info(err.decode())

        cmd = """ ffmpeg -i {} -f image2 -start_number 0 -vf fps=fps=15,scale=iw/4:ih/4:flags=neighbor -qscale:v 2 {}/%05d.{} -loglevel quiet """.format(
            depth_video, depth, "png"
        )
        # print(cmd)

        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            print("Error decoding video from: ", rgb)
            # Write the error to a file
            with open("hoi4d_error.txt", "a") as f:
                f.write(f"{rgb}\n")
        else:
            print(f"Decoded video from: {rgb}")
if __name__ == '__main__':
    root = (
        "/home/wiss/chenh/storage/group/dataset_mirrors/01_incoming/hoi4d/HOI4D_release"
    )
    decode_video(root)
