import yaml
import os

input_parth = r'configs/shanghang/s4dl/s4dl_dann_10_150_250_9.yaml'
with open(input_parth, 'r') as file:
    template = yaml.safe_load(file)
param1_options = [0.1, 0.3, 0.5, 0.7, 0.9]
param2_options = [0.9, 0.7, 0.5, 0.3, 0.1]

# 文件保存路径
save_dir = "configs/shanghang/s4dl/loss_hypers"
os.makedirs(save_dir, exist_ok=True)

# 生成并保存配置文件
for idx1, p1 in enumerate(param1_options):
    for idx2, p2 in enumerate(param2_options):
        config = template.copy()
        config["CRITERION"]["WEIGHTS"][1] = p1
        config["CRITERION"]["WEIGHTS"][2] = p2
        file_name = ("s4dl_dann_10_150_250_9_{}_{}.yaml"
                     .format(int(p1*10), int(p2 * 10)))
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)


