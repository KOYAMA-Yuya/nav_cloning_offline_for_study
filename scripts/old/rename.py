#!/usr/bin/env python3
import os
import roslib

if __name__ == '__main__':
    dir = roslib.packages.get_pkg_dir('nav_cloning') + '/data/img/20250506_00:14:39/'
    base_name = "center"
    new_img_num = 0
    old_img_num = 0

    for new_img_num in range(1695):
        
        for offset_ang in ["-5", "0", "5"]:
            old_name = os.path.join(dir, f"{base_name}{old_img_num}_{offset_ang}.jpg")
            new_name = os.path.join(dir, f"{base_name}{new_img_num}_{offset_ang}.jpg")
            
            if os.path.exists(old_name):  # ファイルが存在するか確認
                os.rename(old_name, new_name)
                print("rename完了")
            else:
                print(f"ファイルが存在しません: {old_name}")

            old_img_num += 1
        