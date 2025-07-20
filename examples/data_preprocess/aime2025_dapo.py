import argparse
import os

import datasets

def main():
    """
    主函数，用于加载、转换和保存AIME 2025数据集。
    """
    parser = argparse.ArgumentParser(
        description="将Hugging Face上的AIME 2025数据集转换为与AIME 2024格式兼容的Parquet文件。"
    )
    # 添加命令行参数，允许用户指定输出目录
    # 默认保存在当前目录下的 'data/aime2025_processed' 文件夹中
    parser.add_argument(
        "--output_dir",
        default="data/aime2025_processed",
        help="保存输出Parquet文件的本地目录。",
    )
    parser.add_argument(
        "--output_filename",
        default="aime2025.parquet",
        help="输出Parquet文件的名称。",
    )
    args = parser.parse_args()

    # --- 1. 定义常量和加载数据集 ---

    # 要加载的数据集名称
    source_dataset_name = "yentinglin/aime_2025"
    # 在新格式中使用的data_source标签
    target_data_source = "AIME2025"

    print(f"正在从Hugging Face Hub加载数据集: {source_dataset_name}...")
    # 加载数据集，'default' configuration
    dataset = datasets.load_dataset(source_dataset_name, "default")
    print("数据集加载成功。")
    print(dataset)

    # --- 2. 定义数据转换函数 ---

    # 定义添加到问题前后的固定文本
    prompt_prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    prompt_suffix = "Remember to put your answer on its own line after \"Answer:\"."

    def format_aime_entry(example, idx):
        """
        将AIME 2025数据集中的单个条目转换为目标格式。

        Args:
            example (dict): 数据集中的一行数据。
            idx (int): 该行数据在原始数据集中的索引。

        Returns:
            dict: 转换为目标格式后的字典。
        """
        # 获取原始问题文本
        raw_problem = example.pop("problem")

        # 构建新的prompt content
        # 格式为：前缀 + 换行 + 问题 + 换行 + 后缀
        prompt_content = f"{prompt_prefix}\n\n{raw_problem}\n\n{prompt_suffix}"

        # 获取答案，它已经是我们需要的格式 (字符串)
        ground_truth_answer = example.pop("answer")

        # 构建新的数据条目
        formatted_data = {
            "data_source": target_data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            "ability": "MATH",  # AIME是数学竞赛，所以能力是MATH
            "reward_model": {
                "style": "rule-lighteval/MATH_v2", # 匹配aime2024的style
                "ground_truth": ground_truth_answer,
            },
            "extra_info": {
                "split": "train",  # 源数据集中只有train分割
                "index": idx,
                "raw_problem": raw_problem,
                # 保留原始数据中的其他有用信息
                "original_id": example.get("id"),
                "url": example.get("url"),
                "year": example.get("year"),
            },
        }
        return formatted_data

    # --- 3. 应用转换并保存结果 ---

    # 我们只处理 'train' 分割，因为这是数据集中唯一存在的部分
    train_dataset = dataset["train"]

    print("正在转换数据集格式...")
    # 使用 .map() 方法应用转换函数到数据集的每一行
    # with_indices=True 会将行索引作为第二个参数(idx)传递给我们的函数
    processed_dataset = train_dataset.map(
        function=format_aime_entry,
        with_indices=True,
        # 移除转换过程中被pop掉或者不再需要的旧列
        remove_columns=train_dataset.column_names
    )
    print("格式转换完成。")
    print("转换后的数据集结构:")
    print(processed_dataset)
    print("\n转换后的第一条数据示例:")
    print(processed_dataset[0])


    # 创建输出目录（如果不存在）
    output_path = os.path.join(args.output_dir, args.output_filename)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n正在将转换后的数据集保存到: {output_path}")
    # 将处理后的数据集保存为Parquet文件
    processed_dataset.to_parquet(output_path)

    print("文件保存成功！脚本执行完毕。")


if __name__ == "__main__":
    main()