import os
import paramiko
import zipfile
from tqdm import tqdm
import time
import shutil
import traceback

# --- 连接配置 (保持不变) ---
conn_dict = {
    "a6000": {
        "hostname": '10.201.1.198',
        "port": 22,
        "username": 'zhoufang',
        "password": 'SEU.student@4A6000'
    },
    "4090": {
        "hostname": "10.128.201.102",
        "port": 22,
        "username": "fangzhou",
        "password": "SEU.student@10gpus"
    }
}

# --- 脚本配置 ---
# 本地源数据所在的根目录 (包含多个要分别打包上传的子目录)
LOCAL_SOURCE_PATH = "/home/zhoufang/.ssh"

# 远程服务器配置键名 (从 conn_dict 中选择)
REMOTE_SERVER_KEY = "4090" # 

# 远程服务器上用于临时存放ZIP文件的路径
# 脚本会在此路径下创建与本地子目录同名的.zip文件
REMOTE_TEMP_ZIP_DIR = "/home/fangzhou/" # 请确保这个目录存在或脚本有权限创建

# 远程服务器上最终解压数据的目标根目录
# 每个子目录的ZIP包会解压到 REMOTE_FINAL_TARGET_DIR/subdir_name/
REMOTE_FINAL_TARGET_DIR = "/home/fangzhou/"

# 是否在解压成功后删除远程服务器上的ZIP文件
DELETE_REMOTE_ZIP_AFTER_UNZIP = True

# 本地临时存放生成的ZIP文件的目录 (脚本运行后会自动清理此目录下的本次zip文件)
LOCAL_TEMP_ZIP_DIR = "./temp_zips_for_upload"
os.makedirs(LOCAL_TEMP_ZIP_DIR, exist_ok=True)


def create_zip_for_subdir(subdir_path, local_temp_zip_dir):
    """
    将指定的子目录打包成一个ZIP文件，存放在本地临时目录中。
    ZIP文件名与子目录名相同。
    """
    subdir_name = os.path.basename(subdir_path)
    local_zip_filename = os.path.join(local_temp_zip_dir, f"{subdir_name}.zip")

    with zipfile.ZipFile(local_zip_filename, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
        for root, _, files in os.walk(subdir_path):
            for file in files:
                file_path = os.path.join(root, file)
                # arcname 是文件在zip包内的相对路径，我们希望它从子目录本身开始
                arcname = os.path.relpath(file_path, os.path.dirname(subdir_path))
                zip_file.write(file_path, arcname)
    print(f"  本地ZIP包创建成功: {local_zip_filename}")
    return local_zip_filename

def upload_file_with_progress(sftp, local_path, remote_path):
    """
    使用 SFTP 上传文件，并显示上传进度条。
    如果远程文件已存在，则跳过。
    """
    try:
        sftp.stat(remote_path)
        print(f"  远程文件 {remote_path} 已存在，跳过上传。")
        return True # 认为已存在也是一种“成功”状态，避免重复操作
    except FileNotFoundError:
        print(f"  远程文件 {remote_path} 不存在，开始上传...")
    except Exception as e:
        print(f"  检查远程文件 {remote_path} 时发生错误: {e}。尝试继续上传...")


    try:
        local_file_size = os.path.getsize(local_path)
        progress_bar = tqdm(
            total=local_file_size, unit="B", unit_scale=True,
            unit_divisor=1024, desc=f"  上传 {os.path.basename(local_path)}"
        )
        def progress_callback(transferred_bytes, _):
            progress_bar.update(transferred_bytes - progress_bar.n)

        sftp.put(local_path, remote_path, callback=progress_callback)
        progress_bar.close()
        print(f"  文件已成功上传到 {remote_path}")
        return True
    except Exception as e:
        if 'progress_bar' in locals() and progress_bar: progress_bar.close()
        print(f"  文件上传失败 ({os.path.basename(local_path)}): {e}")
        traceback.print_exc()
        return False

def ssh_exec_command(ssh_client, command):
    """
    在远程服务器上执行SSH命令并等待完成，返回stdout和stderr。
    """
    print(f"  远程执行命令: {command}")
    try:
        stdin, stdout, stderr = ssh_client.exec_command(command, timeout=300) # 增加超时
        exit_status = stdout.channel.recv_exit_status() # 等待命令执行完成
        stdout_str = stdout.read().decode().strip()
        stderr_str = stderr.read().decode().strip()
        if exit_status == 0:
            if stdout_str: print(f"    远程命令输出:\n{stdout_str}")
            return True, stdout_str, stderr_str
        else:
            print(f"    远程命令执行失败 (退出码: {exit_status})")
            if stdout_str: print(f"    远程命令STDOUT:\n{stdout_str}")
            if stderr_str: print(f"    远程命令STDERR:\n{stderr_str}")
            return False, stdout_str, stderr_str
    except Exception as e:
        print(f"    执行远程命令时出错: {e}")
        traceback.print_exc()
        return False, "", str(e)

def main_process():
    """
    主处理函数。
    """
    server_config = conn_dict.get(REMOTE_SERVER_KEY)
    if not server_config:
        print(f"错误: 未在 conn_dict 中找到服务器配置 '{REMOTE_SERVER_KEY}'")
        return

    ssh = None
    sftp = None
    created_local_zips = [] # 用于跟踪本次创建的本地zip文件，以便清理

    try:
        print(f"开始连接到远程服务器: {server_config['hostname']}...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=server_config['hostname'],
            port=server_config['port'],
            username=server_config['username'],
            password=server_config['password'],
            timeout=10 # 连接超时
        )
        sftp = ssh.open_sftp()
        print("成功连接到远程服务器并打开SFTP通道。")

        # 1. 确保远程临时ZIP目录和最终目标目录存在
        success, _, _ = ssh_exec_command(ssh, f"mkdir -p {REMOTE_TEMP_ZIP_DIR}")
        if not success:
            print(f"无法在远程创建临时ZIP目录 {REMOTE_TEMP_ZIP_DIR}，脚本终止。")
            return
        success, _, _ = ssh_exec_command(ssh, f"mkdir -p {REMOTE_FINAL_TARGET_DIR}")
        if not success:
            print(f"无法在远程创建最终目标目录 {REMOTE_FINAL_TARGET_DIR}，脚本终止。")
            return

        # 2. 获取本地源路径下的所有第一层子目录
        if not os.path.isdir(LOCAL_SOURCE_PATH):
            print(f"错误: 本地源路径 '{LOCAL_SOURCE_PATH}' 不存在或不是一个目录。")
            return
            
        subdirs_to_process = [
            d for d in os.listdir(LOCAL_SOURCE_PATH)
            if os.path.isdir(os.path.join(LOCAL_SOURCE_PATH, d))
        ]
        subdirs_to_process = sorted(subdirs_to_process) # 确保顺序

        if not subdirs_to_process:
            print(f"在 '{LOCAL_SOURCE_PATH}' 下没有找到子目录进行处理。")
            return
            
        print(f"找到 {len(subdirs_to_process)} 个子目录准备处理。")

        for subdir_name in tqdm(subdirs_to_process, desc="处理子目录中"):
            local_subdir_path = os.path.join(LOCAL_SOURCE_PATH, subdir_name)
            print(f"\n正在处理子目录: {local_subdir_path}")

            # a. 在本地打包子目录
            print("  开始本地打包...")
            local_zip_file_path = create_zip_for_subdir(local_subdir_path, LOCAL_TEMP_ZIP_DIR)
            created_local_zips.append(local_zip_file_path)

            # b. 上传ZIP文件到远程服务器的临时目录
            remote_zip_filename = os.path.basename(local_zip_file_path)
            remote_temp_zip_path = os.path.join(REMOTE_TEMP_ZIP_DIR, remote_zip_filename).replace("\\", "/") # 确保是Linux路径分隔符

            print(f"  开始上传 {remote_zip_filename} 到 {remote_temp_zip_path}...")
            if not upload_file_with_progress(sftp, local_zip_file_path, remote_temp_zip_path):
                print(f"  上传 {remote_zip_filename} 失败，跳过此子目录的后续操作。")
                continue # 跳到下一个子目录

            # c. 在远程服务器上解压ZIP文件到最终目标目录
            #    解压后的目录名将是 subdir_name
            remote_target_unzip_path = os.path.join(REMOTE_FINAL_TARGET_DIR, subdir_name).replace("\\", "/")
            
            print(f"  开始在远程服务器上解压 {remote_temp_zip_path} 到 {remote_target_unzip_path}...")
            # 先确保目标解压子目录存在 (unzip -d 不一定会创建多层)
            success, _, _ = ssh_exec_command(ssh, f"mkdir -p {remote_target_unzip_path}")
            if not success:
                 print(f"  无法在远程创建目标解压子目录 {remote_target_unzip_path}，跳过解压。")
                 continue

            # 使用 unzip 命令，-o 表示覆盖已存在文件（如果需要），-d 指定解压目录
            unzip_command = f"unzip -o {remote_temp_zip_path} -d {REMOTE_FINAL_TARGET_DIR}" # 直接解压到目标根，zip包内含子目录名
            # 或者，如果zip包内不含子目录名，而是直接是文件：
            # unzip_command = f"unzip -o {remote_temp_zip_path} -d {remote_target_unzip_path}"

            success, stdout_str, stderr_str = ssh_exec_command(ssh, unzip_command)
            if not success:
                print(f"  远程解压 {remote_zip_filename} 失败。")
                # 可以选择是否继续删除远程ZIP
            else:
                print(f"  远程解压 {remote_zip_filename} 成功。")
                # d. (可选) 删除远程服务器上的ZIP文件
                if DELETE_REMOTE_ZIP_AFTER_UNZIP:
                    print(f"  开始删除远程ZIP文件: {remote_temp_zip_path}")
                    delete_success, _, _ = ssh_exec_command(ssh, f"rm -f {remote_temp_zip_path}")
                    if delete_success:
                        print(f"    远程ZIP文件 {remote_temp_zip_path} 删除成功。")
                    else:
                        print(f"    删除远程ZIP文件 {remote_temp_zip_path} 失败。")
            
            print(f"子目录 {subdir_name} 处理完毕。")

    except paramiko.AuthenticationException:
        print("错误: SSH认证失败！请检查用户名和密码。")
        traceback.print_exc()
    except paramiko.SSHException as sshException:
        print(f"错误: SSH连接出错！错误: {sshException}")
        traceback.print_exc()
    except Exception as e:
        print(f"发生未知错误: {e}")
        traceback.print_exc()
    finally:
        if sftp:
            sftp.close()
            print("SFTP连接已关闭。")
        if ssh:
            ssh.close()
            print("SSH连接已关闭。")
        
        # 清理本地临时生成的ZIP文件
        print("\n开始清理本地临时ZIP文件...")
        for zip_file_to_delete in created_local_zips:
            if os.path.exists(zip_file_to_delete):
                try:
                    os.remove(zip_file_to_delete)
                    print(f"  已删除本地临时ZIP: {zip_file_to_delete}")
                except Exception as e:
                    print(f"  删除本地临时ZIP {zip_file_to_delete} 失败: {e}")
        if os.path.exists(LOCAL_TEMP_ZIP_DIR) and not os.listdir(LOCAL_TEMP_ZIP_DIR): # 如果目录为空则删除
            try:
                os.rmdir(LOCAL_TEMP_ZIP_DIR)
                print(f"  已删除空的本地临时ZIP目录: {LOCAL_TEMP_ZIP_DIR}")
            except Exception as e:
                 print(f"  删除本地临时ZIP目录 {LOCAL_TEMP_ZIP_DIR} 失败: {e}")

        print("脚本执行完毕。")


if __name__ == "__main__":
    # 运行主处理流程
    main_process()